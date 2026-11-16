"""Choose which frames of a patient are shown to a vision-language model.

A VLM sees only a handful of frames per patient, so which ones matters. The
first frames of an angiographic run are usually taken before the contrast
agent has filled the artery; an earlier version sampled evenly over all of a
patient's frames, and with two frames per patient that meant the very first
and the very last frame, the two least informative ones.

The manifest built here takes frames from the middle of a series, spreads
them over different series, and alternates between the right and the left
coronary artery. It is deterministic: the same index gives the same frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

# Where in a series to look, in order of preference: the middle first, then
# the quarter points if more frames are wanted than there are series.
_POSITIONS = (0.5, 0.35, 0.65, 0.25, 0.75)


def _interleave(groups: List[List]) -> List:
    """[[a1, a2, a3], [b1, b2]] -> [a1, b1, a2, b2, a3]"""
    out = []
    for i in range(max((len(g) for g in groups), default=0)):
        out.extend(g[i] for g in groups if i < len(g))
    return out


def build_patient_frame_manifest(
    frames_df: pd.DataFrame,
    n_frames: int = 4,
    id_column: str = "ID",
    path_column: str = "path",
    check_files: bool = True,
) -> Dict[str, List[str]]:
    """Return {patient id (str): [frame paths]} with at most n_frames each.

    Uses the `side`, `series` and `frame_idx` columns written by notebook 03
    when they are present. Without them it falls back to evenly spaced frames
    from the inner part of the patient's frame list.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be at least 1.")
    structured = {"side", "series", "frame_idx"} <= set(frames_df.columns)
    manifest: Dict[str, List[str]] = {}

    for pid, group in frames_df.groupby(id_column, sort=False):
        if check_files:
            group = group[group[path_column].map(lambda p: Path(str(p)).is_file())]
        if group.empty:
            manifest[str(pid)] = []
            continue

        if structured:
            # one list of frames per series, series alternating right / left
            per_side = []
            for _, side_frames in group.groupby("side", sort=True):
                per_side.append([s.sort_values("frame_idx")[path_column].astype(str).tolist()
                                 for _, s in side_frames.groupby("series", sort=True)])
            series_list = _interleave(per_side)
        else:
            paths = group[path_column].astype(str).tolist()
            series_list = [paths]

        chosen: List[str] = []
        for pos in _POSITIONS:
            for frames in series_list:
                pick = frames[min(int(len(frames) * pos), len(frames) - 1)]
                if pick not in chosen:
                    chosen.append(pick)
                if len(chosen) == n_frames:
                    break
            if len(chosen) == n_frames:
                break
        manifest[str(pid)] = chosen

    return manifest


def normalize_patient_id(value) -> str:
    """Patient ids are compared as stripped strings. A trailing '.0' is not
    removed, so an id that was read as a float shows up as a mismatch instead
    of being silently merged with another patient."""
    if pd.isna(value):
        raise ValueError("Patient ID cannot be empty.")
    return str(value).strip()


class PatientFramesDataset:
    """One item = one patient: the sampled frames, the prompt and the label.

    Both VLM adapters use this class, so they are guaranteed to see the same
    frames in the same order.
    """

    def __init__(self, patient_df: pd.DataFrame, frame_manifest: Dict[str, List[str]],
                 prompt: str, prompt_style: str = "default", strict_images: bool = True):
        from PIL import Image  # imported here so the sampling code has no image dependency

        self._Image = Image
        self.patients = patient_df.copy().reset_index(drop=True)
        self.patients["ID"] = self.patients["ID"].map(normalize_patient_id)
        labels = pd.to_numeric(self.patients["has_cad"], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError("has_cad must be 0 or 1.")
        self.patients["has_cad"] = labels.astype(int)
        self.manifest = frame_manifest
        self.prompt = prompt
        self.prompt_style = prompt_style

        missing = [pid for pid in self.patients["ID"] if not self.manifest.get(pid)]
        if missing:
            message = f"{len(missing)} patients have no frames, e.g. {missing[:5]}"
            if strict_images:
                raise ValueError(message)
            import warnings
            warnings.warn(message)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        row = self.patients.iloc[idx]
        pid = row["ID"]
        paths = self.manifest.get(pid, [])
        frames = []
        for p in paths:
            with self._Image.open(p) as img:
                frames.append(img.convert("RGB").copy())
        return {"frames": frames, "prompt": self.prompt, "prompt_style": self.prompt_style,
                "label": int(row["has_cad"]), "ID": pid, "sampled_paths": list(paths)}
