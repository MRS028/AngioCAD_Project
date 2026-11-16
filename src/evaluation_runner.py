"""Zero-shot benchmark of Qwen3-VL and InternVL3 on the AngioCAD patients.

Usage (from anywhere; paths are resolved from this file's location):

    python src/evaluation_runner.py --model qwen       # run one model, cache its answers
    python src/evaluation_runner.py --model internvl
    python src/evaluation_runner.py --model both       # compare, using the caches

An 8 GB GPU cannot hold both models, so each is run on its own and its
patient-level answers are cached in results/. `--model both` runs whatever is
not cached yet and then writes the comparison.

What the comparison guarantees:
  * both models see exactly the same frames (one shared manifest, saved);
  * patients are matched by ID, not by position;
  * a model that gives no usable answer for a patient abstains; abstentions
    are counted and reported, never scored as "no";
  * the head-to-head numbers use the patients both models answered;
  * confidence intervals come from a patient-level bootstrap, and the two
    models' hard predictions are compared with McNemar's test.

Use `--split val` to try prompt variants. The test split should be run once,
with the prompt chosen beforehand.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch

from frame_sampling import build_patient_frame_manifest, normalize_patient_id
from metrics_utils import bootstrap_ci, compute_metrics, empty_metrics, mcnemar_test
from protocol import LABEL_DEFINITION, PROMPT_STYLES

SEED = 42
N_BOOT = 2000
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_DIR = PROJECT_ROOT / "results"
FRAMES_FILE = DATA_DIR / "frames_index.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {"qwen": "Qwen3-VL-8B", "internvl": "InternVL3-8B"}
METRIC_COLS = ["n_total", "n_valid", "n_invalid", "coverage", "accuracy", "auc", "sensitivity",
               "specificity", "precision", "npv", "f1", "balanced_accuracy", "mcc",
               "tp", "tn", "fp", "fn"]


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(p: str) -> str:
    """Frame paths in the index are relative to the project folder; older
    index files written on Windows use backslashes."""
    q = Path(str(p).replace("\\", "/"))
    return str(q if q.is_absolute() else PROJECT_ROOT / q)


def load_patients(split: str) -> pd.DataFrame:
    """Patients of one split with their labels, sorted by ID."""
    path = DATA_DIR / f"clinical_features_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebook 02 first.")
    df = pd.read_csv(path, usecols=["ID", "has_cad"], dtype={"ID": str})
    df["ID"] = df["ID"].map(normalize_patient_id)
    if df["ID"].duplicated().any():
        raise ValueError("Duplicate patient IDs in the split file.")
    if not df["has_cad"].isin([0, 1]).all():
        raise ValueError("has_cad must be 0 or 1.")

    # A patient must not sit in two splits.
    splits_file = DATA_DIR / "splits.json"
    if splits_file.exists():
        with open(splits_file) as f:
            splits = {k: {normalize_patient_id(x) for x in v} for k, v in json.load(f).items()}
        names = list(splits)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if splits[a] & splits[b]:
                    raise ValueError(f"Patients shared between {a} and {b} splits.")
        if set(df["ID"]) != splits.get(split, set(df["ID"])):
            raise ValueError(f"clinical_features_{split}.csv and splits.json disagree.")
    return df.sort_values("ID").reset_index(drop=True)


def check_output(name: str, out: Dict[str, Any], patients: pd.DataFrame) -> None:
    """The model's answers must line up with the patient table, by ID."""
    if list(out["patient_ids"]) != patients["ID"].tolist():
        raise ValueError(f"{name}: patient IDs do not match the patient table.")
    if list(out["labels"]) != patients["has_cad"].astype(int).tolist():
        raise ValueError(f"{name}: labels do not match the patient table.")
    n = len(patients)
    for key in ("predictions", "statuses", "relative_yes_no_scores"):
        if len(out[key]) != n:
            raise ValueError(f"{name}: '{key}' has {len(out[key])} entries, expected {n}.")


def subset_metrics(out: Dict[str, Any], y_true: np.ndarray, idx) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Metrics and CIs over the patients in `idx` that the model answered.

    AUC needs a finite score for every patient it is computed on. If some
    answered patients have no score, AUC (and its CI) is computed on those
    that do, and that number of patients is reported as `n_auc`.
    """
    preds = out["predictions"]
    valid = [int(i) for i in idx if preds[int(i)] is not None]
    if not valid:
        return {**empty_metrics(), "n_auc": 0}, {}

    yt = y_true[valid]
    yp = np.array([preds[i] for i in valid], dtype=int)
    sc = np.array([np.nan if out["relative_yes_no_scores"][i] is None
                   else float(out["relative_yes_no_scores"][i]) for i in valid])
    finite = np.isfinite(sc)

    metrics = compute_metrics(yt, yp)
    ci = bootstrap_ci(yt, yp, n_boot=N_BOOT, seed=SEED)
    if finite.sum() >= 2:
        metrics["auc"] = compute_metrics(yt[finite], yp[finite], sc[finite])["auc"]
        ci["auc"] = bootstrap_ci(yt[finite], yp[finite], sc[finite], n_boot=N_BOOT, seed=SEED)["auc"]
    metrics["n_auc"] = int(finite.sum())
    return metrics, ci


def summarize(out: Dict[str, Any], y_true: np.ndarray, idx, tag: str) -> Dict[str, Any]:
    idx = np.asarray(idx, dtype=int)
    metrics, ci = subset_metrics(out, y_true, idx)
    n_valid = int(sum(out["predictions"][int(i)] is not None for i in idx))
    return {"tag": tag, "n_total": int(len(idx)), "n_valid": n_valid,
            "n_invalid": int(len(idx)) - n_valid,
            "coverage": n_valid / len(idx) if len(idx) else float("nan"),
            **metrics, "ci": ci}


# ---------------------------------------------------------------------------
# Cached per-model answers
# ---------------------------------------------------------------------------
def partial_path(model: str, run_tag: str) -> Path:
    return RESULT_DIR / f"_partial_{model}_{run_tag}.json"


def save_partial(model: str, run_tag: str, out, peak_gb: float, seconds: float, config: dict) -> None:
    path = partial_path(model, run_tag)
    path.write_text(json.dumps({"out": out, "peak_vram_gb": float(peak_gb),
                                "runtime_seconds": float(seconds), "config": config}, allow_nan=True))
    n_none = sum(p is None for p in out["predictions"])
    print(f"[{model}] answers cached in {path.name} ({n_none} abstention(s))")


def load_partial(model: str, run_tag: str, config: dict) -> Dict[str, Any]:
    path = partial_path(model, run_tag)
    payload = json.loads(path.read_text())
    if payload.get("config") != config:
        raise RuntimeError(
            f"{path.name} was produced with different settings ({payload.get('config')}) than "
            f"this run ({config}). Re-run with --force to replace it.")
    out = payload["out"]
    out["predictions"] = [None if v is None else int(v) for v in out["predictions"]]
    out["relative_yes_no_scores"] = [float("nan") if v is None else float(v)
                                     for v in out["relative_yes_no_scores"]]
    out["labels"] = [int(v) for v in out["labels"]]
    return payload


def run_model(model: str, patients: pd.DataFrame, manifest, prompt_style: str):
    """Load one model, score every patient, free the GPU again."""
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    if model == "qwen":
        from qwen3vl_adapter import Qwen3VLSequenceDataset, evaluate_zero_shot, load_qwen3vl
        net, proc = load_qwen3vl(device=DEVICE, load_in_4bit=True)
        out = evaluate_zero_shot(net, proc, Qwen3VLSequenceDataset(patients, manifest, prompt_style=prompt_style), DEVICE)
    else:
        from internvl3_adapter import (InternVL3SequenceDataset, evaluate_zero_shot_internvl,
                                       load_internvl3)
        net, proc = load_internvl3(device=DEVICE, load_in_4bit=True)
        out = evaluate_zero_shot_internvl(net, proc, InternVL3SequenceDataset(patients, manifest, prompt_style=prompt_style), DEVICE)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    seconds = time.time() - start
    del net, proc
    gc.collect()
    torch.cuda.empty_cache()
    return out, peak, seconds


def environment(args) -> Dict[str, Any]:
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
           "seed": SEED, "frames_per_patient": args.frames, "split": args.split,
           "prompt_style": args.prompt_style, "run_mode": args.model}
    for lib in ("transformers", "bitsandbytes", "accelerate"):
        try:
            env[lib] = __import__(lib).__version__
        except ImportError:
            env[lib] = "not installed"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=["qwen", "internvl", "both"], default="both")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--prompt-style", choices=list(PROMPT_STYLES), default="default")
    parser.add_argument("--frames", type=int, default=4, help="frames shown per patient")
    parser.add_argument("--force", action="store_true", help="ignore cached answers and run again")
    args = parser.parse_args()

    if DEVICE.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for the 8B vision-language models.")
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    run_tag = f"{args.split}_{args.prompt_style}"
    suffix = "" if run_tag == "test_default" else f"_{run_tag}"
    config = {"split": args.split, "prompt_style": args.prompt_style, "frames": args.frames}

    print("=" * 78)
    print("AngioCAD zero-shot VLM benchmark")
    print("=" * 78)
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB)")
    print(f"split: {args.split}   prompt: {args.prompt_style}   frames per patient: {args.frames}")

    env = environment(args)
    (RESULT_DIR / f"vlm_environment{suffix}.json").write_text(json.dumps(env, indent=2))

    # ---- patients and frames ------------------------------------------------
    patients = load_patients(args.split)
    y_true = patients["has_cad"].to_numpy(dtype=int)
    print(f"patients: {len(patients)}  (CAD {int(y_true.sum())}, no CAD {int((y_true == 0).sum())})")

    frames = pd.read_csv(FRAMES_FILE, dtype={"ID": str})
    frames["ID"] = frames["ID"].map(normalize_patient_id)
    frames["path"] = frames["path"].map(resolve_path)
    frames = frames[frames["ID"].isin(set(patients["ID"]))]
    manifest = build_patient_frame_manifest(frames, n_frames=args.frames)

    no_frames = [pid for pid in patients["ID"] if not manifest.get(pid)]
    if no_frames:
        # Cannot be scored by an image model; reported, then left out.
        print(f"{len(no_frames)} patient(s) without frames are excluded: {no_frames[:10]}")
        patients = patients[~patients["ID"].isin(no_frames)].reset_index(drop=True)
        y_true = patients["has_cad"].to_numpy(dtype=int)

    pd.DataFrame([{"patient_id": pid, "n_sampled": len(manifest[pid]),
                   "sampled_paths": "|".join(manifest[pid])} for pid in patients["ID"]]
                 ).to_csv(RESULT_DIR / f"vlm_frame_audit{suffix}.csv", index=False)

    # ---- run or load each model ----------------------------------------------
    wanted = list(MODELS) if args.model == "both" else [args.model]
    results: Dict[str, Dict[str, Any]] = {}
    for model in wanted:
        cached = partial_path(model, run_tag)
        if cached.exists() and not args.force:
            print(f"\n[{model}] using cached answers from {cached.name}")
            payload = load_partial(model, run_tag, config)
        else:
            print(f"\n[{model}] loading {MODELS[model]} ...")
            out, peak, seconds = run_model(model, patients, manifest, args.prompt_style)
            check_output(MODELS[model], out, patients)
            save_partial(model, run_tag, out, peak, seconds, config)
            payload = load_partial(model, run_tag, config)
        check_output(MODELS[model], payload["out"], patients)
        preds = payload["out"]["predictions"]
        if all(p is None for p in preds):
            raise RuntimeError(f"[{model}] the model gave no usable answer for any patient. "
                               f"Check the warnings above, then re-run with --force.")
        results[model] = payload

    if args.model != "both":
        print(f"\nDone. Run with --model both (same --split / --prompt-style / --frames) "
              f"once both models are cached to get the comparison.")
        return

    # ---- comparison -------------------------------------------------------------
    q, v = results["qwen"]["out"], results["internvl"]["out"]
    everyone = np.arange(len(patients))
    common = np.array([i for i in everyone if q["predictions"][i] is not None
                       and v["predictions"][i] is not None], dtype=int)

    summaries = {
        ("qwen", "all"): summarize(q, y_true, everyone, "all"),
        ("internvl", "all"): summarize(v, y_true, everyone, "all"),
        ("qwen", "common"): summarize(q, y_true, common, "common_valid"),
        ("internvl", "common"): summarize(v, y_true, common, "common_valid"),
    }
    label = {"all": "all patients, abstentions excluded", "common": "answered by both (primary)"}
    comparison = pd.DataFrame([{"Model": f"{MODELS[m]} ({label[s]})", **{k: summ[k] for k in METRIC_COLS}}
                               for (m, s), summ in summaries.items()])
    comparison.to_csv(RESULT_DIR / f"vlm_baselines_comparison{suffix}.csv", index=False, float_format="%.6f")

    pd.DataFrame({
        "ID": patients["ID"], "y_true": y_true,
        "qwen_status": q["statuses"], "qwen_pred": q["predictions"],
        "qwen_score": q["relative_yes_no_scores"], "qwen_frame_answers": q["generated_outputs"],
        "intern_status": v["statuses"], "intern_pred": v["predictions"],
        "intern_score": v["relative_yes_no_scores"], "intern_frame_answers": v["generated_outputs"],
        "sampled_paths": q["sampled_paths"],
    }).to_csv(RESULT_DIR / f"vlm_predictions_raw{suffix}.csv", index=False, float_format="%.8f")

    if len(common):
        mcnemar = mcnemar_test(y_true[common],
                               np.array([q["predictions"][i] for i in common], dtype=int),
                               np.array([v["predictions"][i] for i in common], dtype=int))
        mcnemar["model_a"], mcnemar["model_b"] = MODELS["qwen"], MODELS["internvl"]
    else:
        mcnemar = {"error": "No patient was answered by both models."}

    details = {
        "environment": env,
        "label_definition": LABEL_DEFINITION,
        "prompt": PROMPT_STYLES[args.prompt_style],
        "n_patients": int(len(patients)),
        "patients_without_frames": no_frames,
        "n_answered_by_both": int(len(common)),
        **{m: {"all": {k: summaries[m, "all"][k] for k in METRIC_COLS},
               "common_valid": {k: summaries[m, "common"][k] for k in METRIC_COLS},
               "ci_common_valid": summaries[m, "common"]["ci"],
               "ci_all": summaries[m, "all"]["ci"],
               "peak_vram_gb": float(results[m]["peak_vram_gb"]),
               "runtime_seconds": float(results[m]["runtime_seconds"])} for m in MODELS},
        "mcnemar_common_valid": mcnemar,
        "notes": [
            "All metrics are at patient level.",
            f"{args.frames} frames per patient from the middle of different series; both models see the same frames.",
            "Patient answer: majority of frame answers, ties broken by the mean yes/no score.",
            "A patient with no usable answer is an abstention and is excluded, not scored as negative.",
            "AUC uses the relative yes/no score of the first generated token; it is not a calibrated probability.",
            f"Bootstrap: {N_BOOT} resamples of patients within each class.",
            "Models are loaded with 4-bit NF4 quantisation (8 GB GPU); full-precision results may differ.",
        ],
    }
    (RESULT_DIR / f"vlm_baselines_details{suffix}.json").write_text(json.dumps(details, indent=2, allow_nan=True))

    # ---- report -------------------------------------------------------------------
    print("\n" + "=" * 78)
    shown = ["Model", "n_total", "n_valid", "coverage", "accuracy", "auc",
             "sensitivity", "specificity", "balanced_accuracy", "mcc"]
    print(comparison[shown].round(3).to_string(index=False))

    print("\n95% bootstrap intervals, patients answered by both models:")
    for m in MODELS:
        print(f"  {MODELS[m]}")
        for metric, c in summaries[m, "common"]["ci"].items():
            if c and np.isfinite(c.get("point", np.nan)):
                print(f"    {metric:18s} {c['point']:.3f} [{c['lower']:.3f}, {c['upper']:.3f}]")
            else:
                print(f"    {metric:18s} undefined")

    print("\nMcNemar test on hard predictions:")
    print(json.dumps(mcnemar, indent=2))
    print(f"\nAn accuracy near the prevalence ({y_true.mean():.3f}) with sensitivity or specificity "
          f"near 0 or 1 means the model is answering the same thing for everyone.")
    print(f"\nFiles written to {RESULT_DIR}")


if __name__ == "__main__":
    main()
