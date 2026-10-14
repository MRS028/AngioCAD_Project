# src/angiocad.py
#  =================================================
# Shared module for AngioCAD project.
# Every notebook does: `from src.angiocad import ...`
# This eliminates copy-paste and prevents "not defined" errors.
#  =================================================

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 224
EXCLUDE_COLS = {"ID", "has_cad", "max_stenosis", "num_diseased_vessels"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def feature_columns(clinical_df: pd.DataFrame) -> list[str]:
    """Return the list of clinical feature columns to be used as model inputs."""
    return [c for c in clinical_df.columns if c not in EXCLUDE_COLS]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------
class BaseAngioDataset(Dataset):
    """Base class with shared image-loading logic."""

    def __init__(self, patient_df, frames_df=None, clinical_df=None,
                 transform=None, aggregate="mean"):
        self.patients = patient_df.reset_index(drop=True)
        self.frames = frames_df
        self.clinical = clinical_df.set_index("ID") if clinical_df is not None else None
        self.transform = transform
        self.aggregate = aggregate
        if self.clinical is not None:
            self.feature_cols = feature_columns(clinical_df)

    def __len__(self) -> int:
        return len(self.patients)

    def _load_frames(self, pid: int) -> list[torch.Tensor]:
        subset = self.frames[self.frames["ID"] == str(int(pid))]
        tensors = []
        for path in subset["path"].tolist():
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(self.transform(img) if self.transform
                               else transforms.ToTensor()(img))
            except Exception:
                continue
        return tensors

    def _aggregate(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        if not tensors:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        stack = torch.stack(tensors)
        if self.aggregate == "max":
            return stack.max(dim=0).values
        if self.aggregate == "first":
            return stack[0]
        return stack.mean(dim=0)


class AngioCADDataset(BaseAngioDataset):
    """Image + clinical table for supervised multimodal training."""

    def __getitem__(self, idx: int):
        row = self.patients.iloc[idx]
        pid = int(row["ID"])
        img = self._aggregate(self._load_frames(pid))
        clinical = self.clinical.loc[pid, self.feature_cols].values.astype(np.float32)
        label = torch.tensor(int(row["has_cad"]), dtype=torch.long)
        return img, torch.from_numpy(clinical), label


class ImageOnlyDataset(BaseAngioDataset):
    """Image-only ablation."""

    def __getitem__(self, idx: int):
        row = self.patients.iloc[idx]
        img = self._aggregate(self._load_frames(int(row["ID"])))
        label = torch.tensor(int(row["has_cad"]), dtype=torch.long)
        return img, label


class ClinicalOnlyDataset(BaseAngioDataset):
    """Clinical-only ablation."""

    def __init__(self, patient_df, clinical_df):
        super().__init__(patient_df, frames_df=None, clinical_df=clinical_df)

    def __getitem__(self, idx: int):
        row = self.patients.iloc[idx]
        pid = int(row["ID"])
        clinical = self.clinical.loc[pid, self.feature_cols].values.astype(np.float32)
        label = torch.tensor(int(row["has_cad"]), dtype=torch.long)
        return torch.from_numpy(clinical), label


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return loss.mean()


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    """ResNet-18 based image encoder for late fusion."""

    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class ClinicalEncoder(nn.Module):
    """MLP encoder for tabular clinical data."""

    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModel(nn.Module):
    """Late-fusion multimodal classifier (image + clinical)."""

    def __init__(self, clinical_dim: int, num_classes: int = 2):
        super().__init__()
        self.img_enc = ImageEncoder(out_dim=256)
        self.clin_enc = ClinicalEncoder(clinical_dim, out_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, image: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        fi = self.img_enc(image)
        fc = self.clin_enc(clinical)
        return self.classifier(torch.cat([fi, fc], dim=1))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_all_data(output_dir: Path = Path("./data/processed")):
    """Load all CSVs and splits in one call."""
    clinical_df = pd.read_csv(output_dir / "clinical_features_merged.csv")
    frames_df = pd.read_csv(output_dir / "frames_index.csv")
    frames_df["ID"] = frames_df["ID"].astype(str)

    with open(output_dir / "splits.json") as f:
        splits = json.load(f)

    class_weights = None
    cw_path = output_dir / "class_weights.json"
    if cw_path.exists():
        with open(cw_path) as f:
            class_weights = json.load(f)

    train_df = clinical_df[clinical_df["ID"].isin(splits["train"])]
    val_df = clinical_df[clinical_df["ID"].isin(splits["val"])]
    test_df = clinical_df[clinical_df["ID"].isin(splits["test"])]

    return clinical_df, frames_df, train_df, val_df, test_df, splits, class_weights