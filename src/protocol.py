"""Task definition shared by the zero-shot VLM adapters.

Both models get the same prompt, their answers are parsed by the same rule,
and frame-level answers are combined into a patient-level answer the same
way. Keeping this in one file is what makes the comparison fair.
"""
from __future__ import annotations

import math
import re
from typing import Literal, Sequence

PredictionStatus = Literal["valid_positive", "valid_negative", "invalid_or_abstained"]

# The prompt used for the reported results.
TASK_PROMPT = (
    "You are analyzing a coronary angiography image for research purposes. "
    "Determine whether there is significant coronary artery stenosis, "
    "defined as more than 50 percent diameter narrowing in a major coronary "
    "artery. Answer with exactly one word: yes or no. "
    "Do not infer disease from image quality or missing anatomy."
)

# A more insistent wording for models that answer "no" to everything. If you
# choose between the two prompts, choose on the validation split and report
# the test split once; picking the better one on the test set is leakage.
TASK_PROMPT_ASSERTIVE = (
    "You are a cardiologist reviewing a coronary angiogram. Examine the "
    "arteries carefully for any narrowing, stenosis, or blockage of more than "
    "50 percent in diameter. If any artery is narrowed by more than 50 percent, "
    "answer yes. If all arteries appear normal or only mildly narrowed, answer no. "
    "Answer with exactly one word: yes or no."
)

PROMPT_STYLES = {"default": TASK_PROMPT, "assertive": TASK_PROMPT_ASSERTIVE}

LABEL_DEFINITION = {
    "positive_class": "CAD present",
    "negative_class": "CAD absent",
    "threshold": "at least one coronary segment graded 51-75% stenosis or worse",
    "note": (
        "Dataset-defined angiographic label from banded segment grades; not a "
        "functional-ischemia or revascularization indication."
    ),
}


def parse_yes_no(text) -> PredictionStatus:
    """Read a one-word answer. Anything that is not clearly yes or no is an
    abstention; it is never turned into a negative."""
    if not isinstance(text, str):
        return "invalid_or_abstained"
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "invalid_or_abstained"
    if words[0] == "yes":
        return "valid_positive"
    if words[0] == "no":
        return "valid_negative"
    return "invalid_or_abstained"


def status_to_label(status: PredictionStatus) -> int | None:
    """1, 0, or None for an abstention."""
    return {"valid_positive": 1, "valid_negative": 0}.get(status)


def aggregate_patient(frame_statuses: Sequence[str], frame_scores: Sequence[float]):
    """Combine the frames of one patient into (status, score).

    The score is the mean of the finite frame scores. The answer is the
    majority of the valid frame answers; a tie is broken by the mean score
    (>= 0.5 means yes). An earlier version abstained on every tie, which with
    two frames per patient threw away each patient whose frames disagreed.
    A patient is an abstention only if no frame gave a usable answer, or the
    frames tie and there is no score to break the tie.
    """
    finite = [s for s in frame_scores if s is not None and math.isfinite(s)]
    score = sum(finite) / len(finite) if finite else float("nan")
    pos = sum(s == "valid_positive" for s in frame_statuses)
    neg = sum(s == "valid_negative" for s in frame_statuses)
    if pos + neg == 0:
        return "invalid_or_abstained", score
    if pos != neg:
        return ("valid_positive" if pos > neg else "valid_negative"), score
    if finite:
        return ("valid_positive" if score >= 0.5 else "valid_negative"), score
    return "invalid_or_abstained", score


# ---------------------------------------------------------------------------
# Scoring the first generated token
# ---------------------------------------------------------------------------
_YES_FORMS = ("yes", "Yes", "YES", " yes", " Yes")
_NO_FORMS = ("no", "No", "NO", " no", " No")


def yes_no_token_ids(tokenizer):
    """Token ids that spell yes / no as a single token.

    Chat models start their answer right after the assistant header, so the
    first token is usually "Yes" or "No" without a leading space. Scoring only
    " yes" / " no" (as an earlier version did) reads the logits of tokens the
    model was never going to produce. All single-token spellings are collected
    and their probabilities are summed.
    """
    def collect(forms):
        ids = []
        for form in forms:
            enc = tokenizer.encode(form, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        return ids

    yes, no = collect(_YES_FORMS), collect(_NO_FORMS)
    ok = bool(yes) and bool(no) and not set(yes) & set(no)
    return yes, no, ok


def relative_yes_score(first_token_logits, yes_ids, no_ids) -> float:
    """P(yes) / (P(yes) + P(no)) from the logits of the first generated token.

    This is a ranking score for ROC analysis, not a calibrated probability.
    """
    import torch

    logits = first_token_logits.float()
    yes = torch.logsumexp(logits[yes_ids], dim=0)
    no = torch.logsumexp(logits[no_ids], dim=0)
    return float(torch.softmax(torch.stack([yes, no]), dim=0)[0].item())
