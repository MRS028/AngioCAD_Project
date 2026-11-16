"""Zero-shot CAD classification with Qwen3-VL.

Each sampled frame is shown to the model with the shared task prompt. The
model generates one token; its text is parsed as yes / no, and the logits of
that token give a relative yes-vs-no score used for ROC analysis. Frame
results are combined per patient by `protocol.aggregate_patient`.

The model is loaded in 4-bit (NF4) so it fits on an 8 GB GPU.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

import torch

from frame_sampling import PatientFramesDataset, normalize_patient_id  # noqa: F401  (re-exported)
from protocol import (PROMPT_STYLES, aggregate_patient, parse_yes_no,
                      relative_yes_score, status_to_label, yes_no_token_ids)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


class Qwen3VLSequenceDataset(PatientFramesDataset):
    """Patient frames with the prompt chosen by name ('default' or 'assertive')."""

    def __init__(self, patient_df, frame_manifest, strict_images=True, prompt_style: str = "default"):
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"Unknown prompt_style {prompt_style!r}; choose from {list(PROMPT_STYLES)}")
        super().__init__(patient_df, frame_manifest, PROMPT_STYLES[prompt_style], prompt_style, strict_images)


def get_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def get_model_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def load_qwen3vl(model_name=DEFAULT_MODEL_NAME, device=None, use_flash_attn=False,
                 device_map="auto", local_files_only=False, load_in_4bit=True,
                 max_gpu_memory_gib: float = 7.0, max_cpu_memory_gib: float = 16.0):
    """Load the model and its processor.

    With `load_in_4bit` the weights are quantised (NF4, double quantisation)
    and a memory cap is set, so layers that do not fit on the GPU are placed
    on the CPU instead of raising an out-of-memory error.
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)

    kwargs: Dict[str, Any] = {"dtype": get_torch_dtype(device), "local_files_only": local_files_only}
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        kwargs["max_memory"] = {0: f"{max_gpu_memory_gib}GiB", "cpu": f"{max_cpu_memory_gib}GiB"}
        kwargs["device_map"] = "auto"
    elif device_map is not None:
        kwargs["device_map"] = device_map

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    except Exception as exc:
        if not use_flash_attn:
            raise
        warnings.warn(f"Flash attention is not available ({exc}); using the default attention.")
        kwargs.pop("attn_implementation", None)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)

    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()
    return model, processor


@torch.inference_mode()
def evaluate_zero_shot(model, processor, dataset, device=None, max_new_tokens: int = 1):
    """Run the model over every patient of `dataset`.

    Returns a dict of equally long lists: statuses, predictions (1 / 0 / None
    for an abstention), relative_yes_no_scores, labels, patient_ids,
    generated_outputs (frame-level answers joined by '|') and sampled_paths.
    """
    model.eval()
    input_device = get_model_input_device(model)
    yes_ids, no_ids, ok = yes_no_token_ids(processor.tokenizer)
    if not ok:
        warnings.warn("Could not find single-token spellings of yes / no for Qwen3-VL; "
                      "scores, and therefore AUC, will be NaN.")

    out = {k: [] for k in ("statuses", "predictions", "relative_yes_no_scores", "labels",
                           "patient_ids", "generated_outputs", "sampled_paths")}
    prompt_style = "default"

    for i in range(len(dataset)):
        sample = dataset[i]
        prompt_style = sample.get("prompt_style", "default")
        frame_statuses, frame_scores = [], []

        for frame in sample["frames"]:
            messages = [{"role": "user", "content": [{"type": "image", "image": frame},
                                                     {"type": "text", "text": sample["prompt"]}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[frame], return_tensors="pt", padding=True).to(input_device)

            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 return_dict_in_generate=True, output_scores=True)
            answer = processor.batch_decode(gen.sequences[:, inputs["input_ids"].shape[1]:],
                                            skip_special_tokens=True)[0]
            frame_statuses.append(parse_yes_no(answer))
            frame_scores.append(relative_yes_score(gen.scores[0][0], yes_ids, no_ids)
                                if ok and gen.scores else float("nan"))

        status, score = aggregate_patient(frame_statuses, frame_scores)
        out["statuses"].append(status)
        out["predictions"].append(status_to_label(status))
        out["relative_yes_no_scores"].append(score)
        out["labels"].append(sample["label"])
        out["patient_ids"].append(sample["ID"])
        out["generated_outputs"].append("|".join(frame_statuses))
        out["sampled_paths"].append("|".join(sample["sampled_paths"]))

        if (i + 1) % 10 == 0 or i + 1 == len(dataset):
            print(f"[Qwen3-VL] {i + 1}/{len(dataset)} patients (prompt: {prompt_style})")

    out["prompt_style"] = prompt_style
    return out
