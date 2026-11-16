"""Metrics for the zero-shot VLM benchmark.

Point estimates, patient-level bootstrap confidence intervals and McNemar's
test. A metric that cannot be computed (for example specificity when there
are no negatives) is reported as NaN, never as zero.
"""
from __future__ import annotations

from math import comb
from typing import Any, Dict, Optional

import numpy as np

CI_METRICS = ("accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy", "mcc", "auc")


def calculate_auc(y_true, scores) -> float:
    """Area under the ROC curve as the Mann-Whitney statistic; ties count half."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(y_true) != len(scores) or not np.isfinite(scores).all():
        return float("nan")
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def compute_metrics(y_true, y_pred, scores=None) -> Dict[str, Any]:
    """Confusion-matrix metrics, plus AUC when continuous scores are given."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length.")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    # F1 = 2TP / (2TP + FP + FN); defined as soon as there is any positive
    # label or prediction, even when precision alone is not.
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    balanced = (recall + specificity) / 2 if np.isfinite(recall) and np.isfinite(specificity) else float("nan")
    mcc = _ratio(tp * tn - fp * fn, float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))

    auc = float("nan")
    if scores is not None:
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(scores) == len(y_true):
            auc = calculate_auc(y_true, scores)

    return {
        "n": int(len(y_true)), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _ratio(tp + tn, len(y_true)),
        "precision": precision, "recall": recall, "sensitivity": recall,
        "specificity": specificity, "npv": _ratio(tn, tn + fn),
        "f1": f1, "balanced_accuracy": balanced, "mcc": mcc, "auc": auc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def empty_metrics() -> Dict[str, Any]:
    """What compute_metrics would return for zero patients."""
    out = compute_metrics(np.array([], dtype=int), np.array([], dtype=int))
    return out


def bootstrap_ci(y_true, y_pred, scores=None, n_boot: int = 2000, seed: int = 42,
                 alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap over patients for the metrics in CI_METRICS.

    Patients are resampled within each class, so every replicate has both
    classes and none has to be thrown away.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = None if scores is None else np.asarray(scores, dtype=float)
    if len(y_true) < 2:
        return {}

    rng = np.random.default_rng(seed)
    point = compute_metrics(y_true, y_pred, scores)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    samples = {k: [] for k in CI_METRICS}
    for _ in range(n_boot):
        parts = [rng.choice(g, len(g)) for g in (pos, neg) if len(g)]
        idx = np.concatenate(parts)
        m = compute_metrics(y_true[idx], y_pred[idx], None if scores is None else scores[idx])
        for k in CI_METRICS:
            samples[k].append(m[k])

    ci = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            ci[k] = {"point": point[k], "lower": float("nan"), "upper": float("nan")}
        else:
            ci[k] = {"point": point[k],
                     "lower": float(np.quantile(arr, alpha / 2)),
                     "upper": float(np.quantile(arr, 1 - alpha / 2))}
    return ci


def mcnemar_test(y_true, pred_a, pred_b) -> Dict[str, Any]:
    """Exact (binomial) McNemar test for two models on the same patients.

    `a_only_correct` and `b_only_correct` are the discordant counts; the
    p-value is two-sided.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=int)
    pred_b = np.asarray(pred_b, dtype=int)
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        return {"error": "Prediction arrays must have equal length."}

    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    a_only = int(np.sum(a_ok & ~b_ok))
    b_only = int(np.sum(~a_ok & b_ok))
    n = a_only + b_only
    if n == 0:
        return {"a_only_correct": 0, "b_only_correct": 0, "n_discordant": 0, "p_value": 1.0,
                "note": "The two models are right and wrong on exactly the same patients."}
    k = min(a_only, b_only)
    p_value = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) * 0.5 ** n)
    return {"a_only_correct": a_only, "b_only_correct": b_only, "n_discordant": n,
            "p_value": float(p_value)}
