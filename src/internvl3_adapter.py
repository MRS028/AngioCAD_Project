"""Zero-shot CAD classification with InternVL3.

Same protocol as the Qwen3-VL adapter: same frames, same prompt, same answer
parsing and the same patient-level aggregation.

Earlier versions asked the model twice per frame, once through `chat()` for
the text and once through `generate()` for the score. The second call fed
the tokenizer a bare "<image>" placeholder, which InternVL does not expand
into image tokens outside `chat()`, so that call failed, the exception was
swallowed, and every score (and the AUC) came out as NaN. Here the prompt is
built the way `chat()` builds it and a single `generate()` call returns both
the answer and the logits.
"""
from __future__ import annotations

import sys
import warnings

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from frame_sampling import PatientFramesDataset, normalize_patient_id  # noqa: F401  (re-exported)
from protocol import (PROMPT_STYLES, aggregate_patient, parse_yes_no,
                      relative_yes_score, status_to_label, yes_no_token_ids)

DEFAULT_MODEL_NAME = "OpenGVLab/InternVL3-8B-Instruct"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"


class InternVL3SequenceDataset(PatientFramesDataset):
    def __init__(self, patient_df, frame_manifest, strict_images=True, prompt_style: str = "default"):
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"Unknown prompt_style {prompt_style!r}; choose from {list(PROMPT_STYLES)}")
        super().__init__(patient_df, frame_manifest, PROMPT_STYLES[prompt_style], prompt_style, strict_images)


def build_transform(input_size: int = 448):
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(image, max_num=6, image_size=448, use_thumbnail=False):
    """Cut an image into square tiles on the grid whose aspect ratio is closest
    to the image's. A square frame gives a single tile."""
    width, height = image.size
    aspect = width / height
    grids = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1)
                    for j in range(1, n + 1) if i * j <= max_num}, key=lambda g: g[0] * g[1])
    cols, rows = min(grids, key=lambda g: abs(aspect - g[0] / g[1]))

    resized = image.resize((image_size * cols, image_size * rows))
    tiles = []
    for k in range(cols * rows):
        left, top = (k % cols) * image_size, (k // cols) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_internvl3(model_name=DEFAULT_MODEL_NAME, device=None, use_flash_attn=False,
                   local_files_only=False, load_in_4bit=True,
                   max_gpu_memory_gib: float = 6.5, max_cpu_memory_gib: float = 16.0):
    """Load InternVL3 and its tokenizer.

    The model's own code on the Hub was written for transformers 4.x and never
    sets `all_tied_weights_keys`, which transformers 5 expects during 4-bit
    loading. The attribute is added while the model loads and the patch is
    removed again afterwards.
    """
    import transformers.modeling_utils as modeling_utils
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_init = modeling_utils.PreTrainedModel.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                              local_files_only=local_files_only, use_fast=False)
    kwargs = {"dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
              "low_cpu_mem_usage": True, "trust_remote_code": True,
              "local_files_only": local_files_only}
    if use_flash_attn:
        kwargs["use_flash_attn"] = True
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        kwargs["max_memory"] = {0: f"{max_gpu_memory_gib}GiB", "cpu": f"{max_cpu_memory_gib}GiB"}
        kwargs["device_map"] = "auto"

    modeling_utils.PreTrainedModel.__init__ = patched_init
    try:
        try:
            model = AutoModel.from_pretrained(model_name, **kwargs)
        except Exception as exc:
            if not use_flash_attn:
                raise
            warnings.warn(f"Flash attention is not available ({exc}); using the default attention.")
            kwargs.pop("use_flash_attn", None)
            model = AutoModel.from_pretrained(model_name, **kwargs)
    finally:
        modeling_utils.PreTrainedModel.__init__ = original_init

    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = {}
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def build_chat_query(model, question: str, num_patches: int) -> tuple[str, str]:
    """The full prompt string that `model.chat()` would build for one image,
    and the separator that ends the assistant's turn."""
    get_conv_template = getattr(sys.modules[type(model).__module__], "get_conv_template")
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], "<image>\n" + question)
    template.append_message(template.roles[1], None)
    image_tokens = IMG_START + IMG_CONTEXT * model.num_image_token * num_patches + IMG_END
    return template.get_prompt().replace("<image>", image_tokens, 1), template.sep.strip()


@torch.inference_mode()
def evaluate_zero_shot_internvl(model, tokenizer, dataset, device=None,
                                max_new_tokens: int = 1, image_size: int = 448, max_num: int = 4):
    """Run the model over every patient of `dataset`; same return value as the
    Qwen3-VL adapter. `max_num` limits the tiles per frame to keep memory low."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    transform = build_transform(image_size)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    yes_ids, no_ids, ok = yes_no_token_ids(tokenizer)
    if not ok:
        warnings.warn("Could not find single-token spellings of yes / no for InternVL3; "
                      "scores, and therefore AUC, will be NaN.")
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT)

    out = {k: [] for k in ("statuses", "predictions", "relative_yes_no_scores", "labels",
                           "patient_ids", "generated_outputs", "sampled_paths")}
    prompt_style = "default"
    n_failed = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        prompt_style = sample.get("prompt_style", "default")
        frame_statuses, frame_scores = [], []

        for frame in sample["frames"]:
            tiles = dynamic_preprocess(frame, max_num=max_num, image_size=image_size)
            pixel_values = torch.stack([transform(t) for t in tiles]).to(device=device, dtype=dtype)
            status, score = "invalid_or_abstained", float("nan")
            try:
                query, sep = build_chat_query(model, sample["prompt"], len(tiles))
                enc = tokenizer(query, return_tensors="pt").to(device)
                gen = model.generate(
                    pixel_values=pixel_values, input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"], max_new_tokens=max_new_tokens,
                    do_sample=False, eos_token_id=tokenizer.convert_tokens_to_ids(sep),
                    return_dict_in_generate=True, output_scores=True)
                # InternVL generates from embeddings, so the sequences hold only new tokens.
                answer = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)[0]
                status = parse_yes_no(answer.split(sep)[0])
                if ok and gen.scores:
                    score = relative_yes_score(gen.scores[0][0], yes_ids, no_ids)
            except Exception as exc:  # one bad frame should not end a long run
                n_failed += 1
                if n_failed <= 3:
                    warnings.warn(f"InternVL3 failed on a frame of patient {sample['ID']}: "
                                  f"{type(exc).__name__}: {exc}")
            frame_statuses.append(status)
            frame_scores.append(score)

        status, score = aggregate_patient(frame_statuses, frame_scores)
        out["statuses"].append(status)
        out["predictions"].append(status_to_label(status))
        out["relative_yes_no_scores"].append(score)
        out["labels"].append(sample["label"])
        out["patient_ids"].append(sample["ID"])
        out["generated_outputs"].append("|".join(frame_statuses))
        out["sampled_paths"].append("|".join(sample["sampled_paths"]))

        if (i + 1) % 10 == 0 or i + 1 == len(dataset):
            print(f"[InternVL3] {i + 1}/{len(dataset)} patients (prompt: {prompt_style})")

    if n_failed:
        print(f"[InternVL3] {n_failed} frame(s) raised an error and were counted as abstentions.")
    out["prompt_style"] = prompt_style
    out["n_failed_frames"] = n_failed
    return out
