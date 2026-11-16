"""Shared code for the AngioCAD notebooks.

Every notebook imports its models, datasets, losses and metrics from here.
Earlier versions copy-pasted these definitions into each notebook, and the
copies drifted apart (different encoders, different checkpoint keys), which
made some checkpoints impossible to reload. Keeping one definition removes
that whole class of error.

Environment switches (useful for a quick dry run on a laptop):
    ANGIOCAD_ROOT=<path>    project folder, if it cannot be found automatically
    ANGIOCAD_PRETRAINED=0   do not download ImageNet weights
    ANGIOCAD_SMOKE=1        tiny epoch counts, two frames per patient, no workers
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, log_loss, matthews_corrcoef, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# --------------------------------------------------------------------------
# Paths and global settings
# --------------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224


def _find_project_root() -> Path:
    """The project folder is the first parent of this file that has a data/ dir."""
    env = os.environ.get("ANGIOCAD_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for p in (here, *here.parents):
        if (p / "data").is_dir():
            return p
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PATIENT_DIR = DATA_DIR / "AngioCAD_Dataset"
FEATURES_FILE = DATA_DIR / "AngioCAD_Features.xlsx"
LABELS_FILE = DATA_DIR / "AngioCAD_Labels.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
FRAMES_DIR = OUTPUT_DIR / "frames"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULT_DIR / "figures"
PRETRAINED = os.environ.get("ANGIOCAD_PRETRAINED", "1") == "1"
SMOKE = os.environ.get("ANGIOCAD_SMOKE", "0") == "1"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, MODEL_DIR, RESULT_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_path(p) -> str:
    """Frame paths are stored relative to the project folder with forward
    slashes, so the index written on Windows also works on Linux or macOS.
    Old index files with backslashes are accepted too."""
    q = Path(str(p).replace("\\", "/"))
    return str(q if q.is_absolute() else PROJECT_ROOT / q)


def relative_path(p) -> str:
    return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()


def environment_info() -> dict:
    """Library versions and hardware, saved next to the results."""
    import sklearn
    import torchvision
    info = dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, torchvision=torchvision.__version__,
                sklearn=sklearn.__version__, numpy=np.__version__,
                pandas=pd.__version__, cuda=torch.version.cuda, seed=SEED)
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------
# Label parsing (used by notebooks 01 and 02, so both agree)
# --------------------------------------------------------------------------
STENOSIS_COLS = [
    "LM", "Prox LAD", "Mid LAD", "Dist LAD", "1st dig", "2nd dig",
    "Prox LCX", "Mid LCX", "Dist LCX", "OM",
    "Prox RCA", "Mid RCA", "Dist RCA", "PDA", "PLB",
]
# Segments are grouped into the four epicardial vessels that clinicians use
# when they talk about 1-, 2- and 3-vessel disease.
VESSEL_GROUPS = {
    "LM": ["LM"],
    "LAD": ["Prox LAD", "Mid LAD", "Dist LAD", "1st dig", "2nd dig"],
    "LCX": ["Prox LCX", "Mid LCX", "Dist LCX", "OM"],
    "RCA": ["Prox RCA", "Mid RCA", "Dist RCA", "PDA", "PLB"],
}
# The spreadsheets grade each segment in bands: NL, 1-25, 26-50, 51-75, 76-90,
# 91-99, 100. A band is represented by its midpoint, so "26-50" becomes 38 and
# "51-75" becomes 63. With a cut-off of 50 the positive class is therefore
# "at least one segment graded 51-75% or worse". A stenosis of exactly 50%
# sits in the 26-50 band and counts as negative; the paper should say so.
CAD_THRESHOLD = 50.0  # percent diameter stenosis


def parse_stenosis(value, unparsed: set | None = None) -> float:
    """Turn a spreadsheet cell into a stenosis percentage.

    'NL' -> 0, '60-70' -> 65, '>90' -> 90, plain numbers pass through.
    Anything else becomes NaN and, if a set is given, is recorded there so the
    caller can see exactly what was dropped instead of losing it silently.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    token = str(value).strip().upper().replace("%", "")
    if token in ("NL", "NORMAL"):
        return 0.0
    try:
        if "-" in token[1:]:
            lo, hi = token.split("-", 1)
            return (float(lo) + float(hi)) / 2.0
        if token[:1] in "<>":
            return float(token[1:])
        return float(token)
    except ValueError:
        if unparsed is not None:
            unparsed.add(str(value))
        return np.nan


def severity_label(max_stenosis: float) -> str:
    if max_stenosis == 0:
        return "Normal"
    if max_stenosis < 30:
        return "Mild"
    if max_stenosis < 50:
        return "Mild-Moderate"
    if max_stenosis < 70:
        return "Moderate"
    return "Severe"


def load_raw_tables():
    """Read the two spreadsheets and tidy the column names.

    Column names are stripped because the features sheet has a 'BUN ' header
    with a trailing space; matching on the exact name silently drops it.
    """
    feats = pd.read_excel(FEATURES_FILE)
    labels = pd.read_excel(LABELS_FILE)
    for df in (feats, labels):
        df.columns = [str(c).strip() for c in df.columns]
        if df["ID"].duplicated().any():
            raise ValueError("Duplicate patient IDs in a raw spreadsheet.")
        df["ID"] = df["ID"].astype(int)
    return feats, labels


def build_labels(labels: pd.DataFrame):
    """Per-patient labels from the segment-level stenosis grades.

    Returns (table, report). Patients with no readable segment at all are
    left out; the report says how many, and lists any cell that could not be
    parsed, so nothing disappears quietly.
    """
    unparsed: set = set()
    seg = pd.DataFrame({c: labels[c].map(lambda v: parse_stenosis(v, unparsed))
                        for c in STENOSIS_COLS if c in labels.columns})
    missing_cols = [c for c in STENOSIS_COLS if c not in labels.columns]
    if missing_cols:
        raise KeyError(f"Stenosis columns missing from the label sheet: {missing_cols}")
    out = pd.DataFrame({"ID": labels["ID"].to_numpy()})
    out["max_stenosis"] = seg.max(axis=1, skipna=True).to_numpy()
    out["n_segments_missing"] = seg.isna().sum(axis=1).to_numpy()
    out["num_diseased_segments"] = seg.ge(CAD_THRESHOLD).sum(axis=1).to_numpy()
    vessel = pd.DataFrame({v: seg[cols].max(axis=1, skipna=True) for v, cols in VESSEL_GROUPS.items()})
    out["num_diseased_vessels"] = vessel.ge(CAD_THRESHOLD).sum(axis=1).to_numpy()
    for v in VESSEL_GROUPS:
        out[f"{v}_max"] = vessel[v].to_numpy()
    keep = out["max_stenosis"].notna()
    report = dict(n_labelled=int(len(out)), n_without_any_grade=int((~keep).sum()),
                  n_with_some_missing_segments=int((out["n_segments_missing"] > 0).sum()),
                  unparsed_values=sorted(unparsed))
    out = out.loc[keep].reset_index(drop=True)
    out["has_cad"] = (out["max_stenosis"] >= CAD_THRESHOLD).astype(int)
    out["severity"] = out["max_stenosis"].map(severity_label)
    return out, report


def build_cohort():
    """Clinical table joined to the labels, one row per patient.

    Both spreadsheets carry the segment grades and the series lists (they
    agree); the copies in the features sheet are dropped so the merged table
    has no _x/_y columns. Returns (cohort, label_report).
    """
    feats, labels = load_raw_tables()
    lab, report = build_labels(labels)
    series_cols = ["Right Coronary Series", "Left Coronary Series"]
    dup = [c for c in STENOSIS_COLS + series_cols if c in feats.columns]
    cohort = (feats.drop(columns=dup)
              .merge(labels[["ID"] + series_cols + STENOSIS_COLS], on="ID", how="inner", validate="one_to_one")
              .merge(lab, on="ID", how="inner", validate="one_to_one"))
    report.update(n_with_clinical=int(len(feats)), n_cohort=int(len(cohort)))
    return cohort, report


# --------------------------------------------------------------------------
# Tabular preprocessing (fitted on the training split only, in notebook 02)
# --------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Age", "Length", "Weight", "BMI", "LVEF", "LVEF dysfunction",
    "TG", "Cholesterol", "HDL", "LDL", "FBS", "Cr", "BUN", "Hgb", "Hct",
    "leukocytes", "Neut", "Monocyte", "MXD", "Lymph", "PLT", "RDW",
]
CATEGORICAL_FEATURES = [
    "Sex", "dominance", "DM", "HTN", "Hyperlipidemia", "Current Smoker",
    "PVD", "Stroke", "Beta blocker", "Alpha blocker", "Diuretic",
    "ACEI", "ARB", "CCB", "ASA", "Clopidogrel", "Astatine",
]
# Deliberately not used as inputs:
#   Gensini Score - computed from the angiogram, i.e. from the label itself.
#   Diagnosis, IHD - describe the presentation/known disease; kept for
#                    subgroup analysis only.
LEAKY_COLUMNS = ["Gensini Score"]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data.

    The lab values contain a few entries that are clearly unit or typing
    errors (BMI 148, haemoglobin 129, creatinine 23). Left alone they stretch
    the scaler and dominate the standardised features.
    """

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric = Pipeline([("clip", QuantileClipper()),
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # one column per binary variable instead of two mirrored ones
        ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, list(numeric_cols)),
                              ("cat", categorical, list(categorical_cols))])


def clean_feature_name(name: str) -> str:
    """'num__Age' -> 'Age', 'cat__DM_1' -> 'DM', 'cat__Sex_Male' -> 'Sex_Male'."""
    name = re.sub(r"^(num|cat)__", "", name)
    return re.sub(r"_1(\.0)?$", "", name)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data() -> SimpleNamespace:
    """Load the leakage-free splits written by notebook 02 and 03."""
    d = SimpleNamespace()
    d.train, d.val, d.test = (
        pd.read_csv(OUTPUT_DIR / f"clinical_features_{s}.csv").reset_index(drop=True)
        for s in ("train", "val", "test")
    )
    for df in (d.train, d.val, d.test):
        df["ID"] = df["ID"].astype(int)
    d.frames = pd.read_csv(OUTPUT_DIR / "frames_index.csv")
    d.frames["ID"] = d.frames["ID"].astype(int)  # ids are ints everywhere
    d.frames["path"] = d.frames["path"].map(resolve_path)
    if SMOKE:
        d.frames = d.frames.groupby("ID").head(2).reset_index(drop=True)
    d.original = pd.read_csv(OUTPUT_DIR / "clinical_features_original.csv")
    d.original["ID"] = d.original["ID"].astype(int)
    d.metadata = pd.read_csv(OUTPUT_DIR / "clinical_metadata.csv")
    d.metadata["ID"] = d.metadata["ID"].astype(int)
    with open(OUTPUT_DIR / "feature_dim.json") as f:
        meta = json.load(f)
    d.feature_cols = list(meta["feature_names_out"])
    d.clinical_dim = len(d.feature_cols)
    found = set(d.train.columns) - {"ID", "has_cad"}
    if found != set(d.feature_cols):
        raise ValueError(
            "feature_dim.json and the training CSV disagree.\n"
            f"  only in json: {sorted(set(d.feature_cols) - found)}\n"
            f"  only in csv : {sorted(found - set(d.feature_cols))}"
        )
    ids = {s: set(getattr(d, s)["ID"]) for s in ("train", "val", "test")}
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise ValueError("The same patient appears in more than one split.")
    with open(OUTPUT_DIR / "class_weights.json") as f:
        cw = json.load(f)
    d.class_weights = torch.tensor([cw["0"], cw["1"]], dtype=torch.float32)
    # Fingerprint of the splits and features. Checkpoints remember it, so a
    # model trained before notebook 02 was re-run is never reused by mistake.
    blob = json.dumps([sorted(ids[s]) for s in ("train", "val", "test")] + [d.feature_cols])
    d.key = hashlib.md5(blob.encode()).hexdigest()[:12]
    return d


def get_transforms(train: bool) -> transforms.Compose:
    """Image transforms.

    No horizontal flip: left/right orientation carries anatomical meaning in
    angiography (LAD vs RCA views), so flipping can change what the picture
    means. Mild rotation, translation and brightness jitter only.
    """
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + norm)
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ] + norm)


class FrameDataset(Dataset):
    """One sample = one angiographic frame (multiple-instance setting).

    The patient's label is attached to every frame, and the patient id is
    returned so predictions can be pooled back to one score per patient.
    """

    def __init__(self, patient_df, frames_df, feature_cols, transform=None,
                 use_clinical=True):
        pdf = patient_df[["ID", "has_cad"]].copy()
        pdf["ID"] = pdf["ID"].astype(int)
        fr = frames_df.copy()
        fr["ID"] = fr["ID"].astype(int)
        self.frames = fr.merge(pdf, on="ID", how="inner").reset_index(drop=True)
        self.paths = self.frames["path"].tolist()
        self.pids = self.frames["ID"].to_numpy()
        self.labels = self.frames["has_cad"].astype(int).to_numpy()
        self.transform = transform
        self.use_clinical = use_clinical
        self.missing_patients = sorted(set(pdf["ID"]) - set(self.pids))
        if use_clinical:
            clin = patient_df.set_index("ID")[feature_cols].astype(np.float32)
            self._clin = {int(i): r.to_numpy().copy() for i, r in clin.iterrows()}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = torch.tensor(int(self.labels[i]), dtype=torch.long)
        pid = int(self.pids[i])
        if self.use_clinical:
            return img, torch.from_numpy(self._clin[pid]), y, pid
        return img, y, pid


class ClinicalDataset(Dataset):
    """One sample = one patient, tabular features only."""

    def __init__(self, patient_df, feature_cols):
        self.x = patient_df[feature_cols].to_numpy(dtype=np.float32)
        self.y = patient_df["has_cad"].astype(int).to_numpy()
        self.pids = patient_df["ID"].astype(int).to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.x[i]),
                torch.tensor(int(self.y[i]), dtype=torch.long),
                int(self.pids[i]))


def make_balanced_sampler(ds: FrameDataset, class_balance: bool = True,
                          seed: int = SEED) -> WeightedRandomSampler:
    """Sample frames so every patient counts equally, and (optionally) so
    the two classes get equal total weight.

    Without this, a patient with 80 frames pulls eight times harder on the
    gradient than one with 10, and the majority class dominates.
    """
    pids = pd.Series(ds.pids)
    w = 1.0 / pids.map(pids.value_counts()).to_numpy()
    if class_balance:
        per_patient = pd.DataFrame({"pid": ds.pids, "y": ds.labels}).drop_duplicates("pid")
        class_n = per_patient["y"].value_counts()
        w = w / pd.Series(ds.labels).map(class_n).to_numpy()
    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(w), replacement=True, generator=gen)


MODE_OF = {"fusion": "fusion", "cross_attention": "fusion",
           "image": "image", "clinical": "clinical"}


def make_loaders(data, cfg, kind: str, seed: int = SEED) -> dict:
    mode = MODE_OF[kind]
    nw = cfg["num_workers"]
    common = dict(batch_size=cfg["batch_size"], num_workers=nw,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=nw > 0, worker_init_fn=_seed_worker)
    gen = torch.Generator().manual_seed(seed)
    out = {}
    if mode == "clinical":
        sets = {k: ClinicalDataset(getattr(data, k), data.feature_cols)
                for k in ("train", "val", "test")}
        if cfg["balanced_sampler"]:
            # same idea as the frame sampler: both classes get equal weight
            y = pd.Series(sets["train"].y)
            w = (1.0 / y.map(y.value_counts())).to_numpy().copy()
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(w),
                                            replacement=True, generator=gen)
            out["train"] = DataLoader(sets["train"], sampler=sampler, **common)
        else:
            out["train"] = DataLoader(sets["train"], shuffle=True, generator=gen, **common)
        out["val"] = DataLoader(sets["val"], shuffle=False, **common)
        out["test"] = DataLoader(sets["test"], shuffle=False, **common)
        out["sets"] = sets
        return out
    use_clin = mode == "fusion"
    sets = {
        "train": FrameDataset(data.train, data.frames, data.feature_cols,
                              get_transforms(True), use_clin),
        "val": FrameDataset(data.val, data.frames, data.feature_cols,
                            get_transforms(False), use_clin),
        "test": FrameDataset(data.test, data.frames, data.feature_cols,
                             get_transforms(False), use_clin),
    }
    if cfg["balanced_sampler"]:
        sampler = make_balanced_sampler(sets["train"], True, seed)
        out["train"] = DataLoader(sets["train"], sampler=sampler, **common)
    else:
        out["train"] = DataLoader(sets["train"], shuffle=True, generator=gen, **common)
    out["val"] = DataLoader(sets["val"], shuffle=False, **common)
    out["test"] = DataLoader(sets["test"], shuffle=False, **common)
    out["sets"] = sets
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    """ResNet-18 backbone followed by a small projection.

    The stem and first residual stage are frozen: with ~250 training patients
    there is not enough signal to retrain generic edge filters, and freezing
    them cuts the number of trainable parameters.
    """

    def __init__(self, out_dim=256, pretrained=True, dropout=0.5, freeze_stem=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.freeze_stem = freeze_stem
        if freeze_stem:
            for m in (backbone.conv1, backbone.bn1, backbone.layer1):
                for p in m.parameters():
                    p.requires_grad = False
        self.proj = nn.Sequential(nn.Linear(feat_dim, out_dim), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout))
        self.out_dim = out_dim

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_stem:  # keep frozen BatchNorm statistics fixed
            self.backbone.bn1.eval()
            self.backbone.layer1.eval()
        return self

    def forward(self, x):
        return self.proj(self.backbone(x))


class ClinicalEncoder(nn.Module):
    """MLP for the tabular vector.

    LayerNorm instead of BatchNorm: in the frame-level setting every frame of
    a patient carries the same clinical vector, so batch statistics would mix
    information across frames. Gaussian input noise during training stops the
    network from memorising individual patients through their clinical vector.
    """

    def __init__(self, in_dim, out_dim=128, dropout=0.5, noise=0.1):
        super().__init__()
        self.noise = noise
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, out_dim), nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        if self.training and self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return self.net(x)


def _head(in_dim, num_classes=2, dropout=0.5):
    return nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
                         nn.Dropout(dropout), nn.Linear(64, num_classes))


class ImageOnlyModel(nn.Module):
    def __init__(self, clinical_dim=None, pretrained=True, dropout=0.5, **_):
        super().__init__()
        self.img_enc = ImageEncoder(256, pretrained, dropout)
        self.head = _head(256, dropout=dropout)

    def forward(self, image):
        return self.head(self.img_enc(image))


class ClinicalOnlyModel(nn.Module):
    def __init__(self, clinical_dim, dropout=0.5, clinical_noise=0.1, **_):
        super().__init__()
        self.clin_enc = ClinicalEncoder(clinical_dim, 128, dropout, clinical_noise)
        self.head = _head(128, dropout=dropout)

    def forward(self, clinical):
        return self.head(self.clin_enc(clinical))


class FusionModel(nn.Module):
    """Late fusion: concatenate image and clinical embeddings.

    Modality dropout randomly blanks one branch during training so the model
    cannot rely on a single modality (and, in particular, cannot lean entirely
    on the clinical vector that is repeated across frames).
    """

    def __init__(self, clinical_dim, pretrained=True, dropout=0.5,
                 clinical_noise=0.1, modality_dropout=0.2, **_):
        super().__init__()
        self.img_enc = ImageEncoder(256, pretrained, dropout)
        self.clin_enc = ClinicalEncoder(clinical_dim, 128, dropout, clinical_noise)
        self.head = _head(256 + 128, dropout=dropout)
        self.modality_dropout = modality_dropout

    def forward(self, image, clinical):
        zi, zc = self.img_enc(image), self.clin_enc(clinical)
        if self.training and self.modality_dropout > 0:
            r = torch.rand(zi.size(0), device=zi.device)
            keep_i = (r >= self.modality_dropout / 2).float().unsqueeze(1)
            keep_c = ((r < self.modality_dropout / 2) | (r >= self.modality_dropout)).float().unsqueeze(1)
            zi, zc = zi * keep_i, zc * keep_c
        return self.head(torch.cat([zi, zc], dim=1))


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q, self.norm_kv, self.norm_out = (nn.LayerNorm(d_model) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))

    def forward(self, query, key_value):
        out, w = self.attn(self.norm_q(query), self.norm_kv(key_value),
                           self.norm_kv(key_value), need_weights=True)
        x = query + out
        return x + self.ffn(self.norm_out(x)), w


class CrossAttentionFusionModel(nn.Module):
    """Image tokens and one-token-per-clinical-feature attend to each other.

    Kept deliberately small (d_model 128): with a few hundred patients a wide
    transformer has far more capacity than data.
    """

    def __init__(self, clinical_dim, pretrained=True, d_model=128, n_heads=4,
                 n_layers=2, n_img_tokens=4, dropout=0.2, clinical_noise=0.1, **_):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.img_proj = nn.Linear(feat_dim, n_img_tokens * d_model)
        self.n_img_tokens, self.d_model = n_img_tokens, d_model
        self.feat_proj = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, d_model))
        self.feat_embed = nn.Parameter(torch.randn(1, clinical_dim, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.clin_norm = nn.LayerNorm(d_model)
        self.i2c = nn.ModuleList(CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers))
        self.c2i = nn.ModuleList(CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers))
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, 64),
                                  nn.GELU(), nn.Dropout(0.5), nn.Linear(64, 2))
        self.clinical_noise = clinical_noise

    def forward(self, image, clinical, return_attention=False):
        if self.training and self.clinical_noise > 0:
            clinical = clinical + self.clinical_noise * torch.randn_like(clinical)
        img = self.img_proj(self.backbone(image)).view(-1, self.n_img_tokens, self.d_model)
        clin = self.feat_proj(clinical.unsqueeze(-1)) + self.feat_embed
        clin = self.clin_norm(torch.cat([self.cls.expand(clin.size(0), -1, -1), clin], dim=1))
        maps = []
        for a, b in zip(self.i2c, self.c2i):
            img, w = a(img, clin)
            clin, _ = b(clin, img)
            maps.append(w.detach().cpu())
        logits = self.head(torch.cat([img.mean(1), clin[:, 0]], dim=1))
        return (logits, maps) if return_attention else logits


def build_model(kind: str, clinical_dim: int, cfg: dict) -> nn.Module:
    kw = dict(clinical_dim=clinical_dim, pretrained=cfg["pretrained"],
              clinical_noise=cfg["clinical_noise"],
              modality_dropout=cfg["modality_dropout"])
    cls = {"fusion": FusionModel, "image": ImageOnlyModel,
           "clinical": ClinicalOnlyModel, "cross_attention": CrossAttentionFusionModel}[kind]
    return cls(**kw)


def load_model(ckpt_path, device):
    """Rebuild a model from the architecture recorded in the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ckpt["cfg"].copy()
    cfg["pretrained"] = False  # weights come from the checkpoint
    model = build_model(ckpt["model_kind"], ckpt["clinical_dim"], cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Multi-class focal loss, FL = (1 - p_t)^gamma * CE.

    p_t is the model's probability for the true class, computed from the
    unweighted, unsmoothed softmax. Class weights multiply the final loss once.
    (A common mistake is to take p_t = exp(-weighted_ce); that makes the
    focusing term depend on the class weight and double counts the weighting.)
    With gamma = 0 and no weights this is exactly cross-entropy.
    """

    def __init__(self, gamma=2.0, class_weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma, self.label_smoothing = gamma, label_smoothing
        if class_weight is not None:
            self.register_buffer("class_weight", class_weight.float())
        else:
            self.class_weight = None

    def forward(self, logits, targets):
        p_t = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        loss = (1.0 - p_t).clamp(min=0) ** self.gamma * ce
        if self.class_weight is not None:
            loss = loss * self.class_weight[targets]
        return loss.mean()


def make_criterion(cfg, class_weights, device):
    w = class_weights.to(device) if cfg["use_class_weights"] else None
    if cfg["loss"] == "ce":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])
    return FocalLoss(cfg["focal_gamma"], w, cfg["label_smoothing"])


# --------------------------------------------------------------------------
# Training and prediction
# --------------------------------------------------------------------------
def get_cfg(**overrides) -> dict:
    """Default training recipe shared by every deep model in the study.

    Using one recipe for the image-only, clinical-only and fusion models keeps
    the ablation honest: any difference comes from the inputs, not the tuning.
    """
    cfg = dict(
        epochs=25, patience=6, batch_size=32, lr=2e-4, backbone_lr_mult=0.1,
        weight_decay=1e-2, grad_clip=5.0, loss="focal", focal_gamma=2.0,
        label_smoothing=0.05, use_class_weights=False, balanced_sampler=True,
        clinical_noise=0.1, modality_dropout=0.2, amp=True,
        num_workers=0 if os.name == "nt" else 4, pretrained=PRETRAINED,
    )
    if SMOKE:
        cfg.update(epochs=2, patience=2, num_workers=0, batch_size=8)
    cfg.update(overrides)
    return cfg


def unpack(batch, mode, device):
    if mode == "fusion":
        img, clin, y, pid = batch
        inputs = (img.to(device, non_blocking=True), clin.to(device, non_blocking=True))
    elif mode == "image":
        img, y, pid = batch
        inputs = (img.to(device, non_blocking=True),)
    else:
        clin, y, pid = batch
        inputs = (clin.to(device, non_blocking=True),)
    return inputs, y.to(device, non_blocking=True), pid


def train_one_epoch(model, loader, criterion, optimizer, device, mode, grad_clip,
                    scaler=None):
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    total, n = 0.0, 0
    for batch in loader:
        inputs, y, _ = unpack(batch, mode, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(*inputs)
        loss = criterion(logits.float(), y)  # the loss itself stays in float32
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


@torch.no_grad()
def predict_patients(model, loader, device, mode, pooling="mean"):
    """Pool frame-level probabilities into one score per patient.

    Returns (patient_ids, y_true, y_prob) with ids sorted, so the three arrays
    are aligned and can be joined to other tables on the id.
    """
    model.eval()
    probs, labels = defaultdict(list), {}
    for batch in loader:
        inputs, y, pid = unpack(batch, mode, device)
        p = torch.softmax(model(*inputs).float(), dim=1)[:, 1].cpu().numpy()
        for pi, yi, idi in zip(p, y.cpu().numpy(), pid.numpy()):
            probs[int(idi)].append(float(pi))
            labels[int(idi)] = int(yi)
    pids = np.array(sorted(probs))
    agg = {"mean": np.mean, "max": np.max, "median": np.median}[pooling]
    return (pids, np.array([labels[i] for i in pids]),
            np.array([agg(probs[i]) for i in pids]))


def fit(model, loaders, kind, criterion, device, cfg, ckpt_path, clinical_dim,
        seed=SEED, verbose=True, data_key=None):
    """Train with early stopping on patient-level validation AUC.

    Ctrl-C stops training cleanly and keeps the best checkpoint so far.
    """
    mode = MODE_OF[kind]
    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    groups = [{"params": rest, "lr": cfg["lr"]}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg["lr"] * cfg["backbone_lr_mult"]})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-7)

    scaler = torch.amp.GradScaler(device.type, enabled=bool(cfg.get("amp")) and device.type == "cuda")

    history, best_auc, best_epoch, best_state, wait = [], -1.0, 0, None, 0
    interrupted = False
    if verbose:
        print(f"{'epoch':>5} {'train_loss':>11} {'val_auc':>8} {'val_logloss':>12}")
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            tr = train_one_epoch(model, loaders["train"], criterion, opt, device, mode,
                                 cfg["grad_clip"], scaler)
            sched.step()
            _, yt, yp = predict_patients(model, loaders["val"], device, mode)
            auc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
            ll = float(log_loss(yt, np.clip(yp, 1e-6, 1 - 1e-6), labels=[0, 1]))
            history.append(dict(epoch=epoch, train_loss=tr, val_auc=auc, val_logloss=ll))
            improved = auc > best_auc + 1e-4
            if verbose:
                print(f"{epoch:>5} {tr:>11.4f} {auc:>8.4f} {ll:>12.4f}{'  *' if improved else ''}")
            if improved:
                best_auc, best_epoch, wait = auc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(dict(model_state=best_state, epoch=epoch, best_val_auc=best_auc,
                                model_kind=kind, clinical_dim=int(clinical_dim), cfg=cfg,
                                seed=int(seed), data_key=data_key, finished=False), ckpt_path)
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    if verbose:
                        print(f"stopping: no val AUC gain for {cfg['patience']} epochs")
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted; keeping the best checkpoint so far "
              "(it is marked unfinished, so the next run trains again)")
    if best_state is not None:
        model.load_state_dict(best_state)
        if not interrupted:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ckpt["finished"] = True
            torch.save(ckpt, ckpt_path)
    return pd.DataFrame(history), dict(best_epoch=best_epoch, best_val_auc=best_auc,
                                       finished=not interrupted)


def _same_recipe(a: dict, b: dict) -> bool:
    skip = {"num_workers"}  # does not change the result
    return {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}


def run_experiment(data, kind, cfg, seed, tag=None, device=None, verbose=True, reuse=True):
    """Train one model and return its validation and test predictions.

    The checkpoint goes to models/<tag>_seed<seed>.pt (tag defaults to the
    model kind). If a finished checkpoint with the same recipe, seed and data
    already exists it is loaded instead of training again, so notebooks 06-10
    can share the model trained in notebook 05 and an interrupted session can
    simply be re-run. Pass reuse=False to force a fresh run.
    """
    device = device or get_device()
    ensure_dirs()
    name = tag or kind
    ckpt_path = MODEL_DIR / f"{name}_seed{seed}.pt"
    hist_path = MODEL_DIR / f"{name}_seed{seed}_history.csv"
    mode = MODE_OF[kind]

    seed_everything(seed)
    loaders = make_loaders(data, cfg, kind, seed)
    model, hist, info, reused = None, None, None, False
    if reuse and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if (ckpt.get("finished") and ckpt.get("model_kind") == kind
                and ckpt.get("data_key") == data.key and _same_recipe(ckpt["cfg"], cfg)):
            model, _ = load_model(ckpt_path, device)
            hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
            info = dict(best_epoch=int(ckpt["epoch"]), best_val_auc=float(ckpt["best_val_auc"]),
                        finished=True)
            reused = True
            if verbose:
                print(f"reusing {ckpt_path.name} (best epoch {info['best_epoch']}, "
                      f"val AUC {info['best_val_auc']:.4f})")
    if model is None:
        model = build_model(kind, data.clinical_dim, cfg).to(device)
        criterion = make_criterion(cfg, data.class_weights, device)
        hist, info = fit(model, loaders, kind, criterion, device, cfg, ckpt_path,
                         data.clinical_dim, seed, verbose, data.key)
        hist.to_csv(hist_path, index=False)
    val = predict_patients(model, loaders["val"], device, mode)
    test = predict_patients(model, loaders["test"], device, mode)
    return dict(model=model, history=hist, info=info, val=val, test=test, loaders=loaders,
                ckpt_path=ckpt_path, seed=seed, kind=kind, name=name, reused=reused)


# --------------------------------------------------------------------------
# Metrics and statistics
# --------------------------------------------------------------------------
def auc_safe(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def threshold_metrics(y_true, y_prob, thr):
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")
    return dict(
        threshold=float(thr), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        accuracy=div(tp + tn, tp + tn + fp + fn),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        sensitivity=div(tp, tp + fn), specificity=div(tn, tn + fp),
        precision=div(tp, tp + fp), npv=div(tn, tn + fn),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
    )


def tune_threshold(y_true, y_prob):
    """Pick the threshold that maximises Youden's J (sensitivity + specificity - 1).

    Use this on validation data only. F1 is a poor target here: with 78%
    positives, calling everyone positive already gives F1 = 0.88.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -np.inf
    i = int(np.argmax(j))
    return float(thr[i]), float(j[i])


def bootstrap_ci(y_true, y_prob, fn, n_boot=2000, alpha=0.05, seed=SEED):
    """Percentile bootstrap, resampling patients within each class so that
    every replicate contains both classes. Returns (point, low, high)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v = fn(y_true[idx], y_prob[idx])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(fn(y_true, y_prob)), float(lo), float(hi)


def paired_auc_difference(y_true, p_a, p_b, n_boot=5000, seed=SEED):
    """Paired bootstrap for AUC(a) - AUC(b) on the same patients.

    Returns (difference, low, high, two-sided p). A CI that straddles zero
    means the data cannot separate the two models.
    """
    y_true, p_a, p_b = map(np.asarray, (y_true, p_a, p_b))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs.append(roc_auc_score(y_true[idx], p_a[idx]) - roc_auc_score(y_true[idx], p_b[idx]))
    diffs = np.array(diffs)
    # a bootstrap p-value cannot be smaller than one over the number of resamples
    p = min(1.0, max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0 / n_boot))
    d = roc_auc_score(y_true, p_a) - roc_auc_score(y_true, p_b)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(d), float(lo), float(hi), float(p)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        hi_closed = i == n_bins - 1
        m = (y_prob >= edges[i]) & ((y_prob <= edges[i + 1]) if hi_closed else (y_prob < edges[i + 1]))
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def summarise(name, role, pids, y_true, y_prob, threshold, seed=None, n_boot=2000,
              save=True, extra=None):
    """Compute the standard metric block with 95% CIs and save it.

    Writes results/predictions_<name>.csv and results/summary_<name>.json so
    later notebooks can compare models on exactly the same patients.
    """
    ci = lambda fn: bootstrap_ci(y_true, y_prob, fn, n_boot)
    at = lambda key: (lambda yt, yp: threshold_metrics(yt, yp, threshold)[key])
    out = dict(
        model=name, role=role, seed=seed, n_patients=int(len(y_true)),
        prevalence=float(np.mean(y_true)), threshold=float(threshold),
        auc=ci(auc_safe),
        auprc=ci(lambda yt, yp: float(average_precision_score(yt, yp))),
        balanced_accuracy=ci(at("balanced_accuracy")),
        sensitivity=ci(at("sensitivity")), specificity=ci(at("specificity")),
        f1=ci(at("f1")), mcc=ci(at("mcc")),
        at_threshold=threshold_metrics(y_true, y_prob, threshold),
    )
    out.update(extra or {})
    if save:
        ensure_dirs()
        pd.DataFrame(dict(ID=pids, y_true=y_true, y_prob=y_prob)).to_csv(
            RESULT_DIR / f"predictions_{name}.csv", index=False)
        with open(RESULT_DIR / f"summary_{name}.json", "w") as f:
            json.dump(out, f, indent=2)
    return out


def fit_platt(y_val, p_val):
    """Platt scaling fitted on validation predictions.

    Training with a class-balanced sampler shifts the predicted probabilities
    away from the true prevalence, so raw outputs rank patients well but are
    not calibrated. Returns a function that maps raw to calibrated scores.
    """
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logit(np.asarray(p_val)).reshape(-1, 1), np.asarray(y_val))
    return lambda p: lr.predict_proba(logit(np.asarray(p)).reshape(-1, 1))[:, 1]


def load_predictions(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULT_DIR / f"predictions_{name}.csv").sort_values("ID").reset_index(drop=True)


def collect_summaries() -> pd.DataFrame:
    """One row per saved model summary, for the final comparison table."""
    rows = []
    for path in sorted(RESULT_DIR.glob("summary_*.json")):
        with open(path) as f:
            s = json.load(f)
        row = dict(model=s["model"], role=s.get("role"), n=s["n_patients"], threshold=s["threshold"])
        for k in ("auc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "f1", "mcc"):
            row[k] = s[k][0]
            row[f"{k}_ci"] = format_ci(s[k])
        rows.append(row)
    return pd.DataFrame(rows)


def format_ci(t, digits=3):
    return f"{t[0]:.{digits}f} [{t[1]:.{digits}f}, {t[2]:.{digits}f}]"


def no_skill_reference(y_true):
    """What a classifier that always predicts 'CAD' would score."""
    prev = float(np.mean(y_true))
    return dict(accuracy=prev, f1=2 * prev / (1 + prev), balanced_accuracy=0.5, auc=0.5)


# --------------------------------------------------------------------------
# Text prompts for the CLIP baseline
# --------------------------------------------------------------------------
def _yes_no(v):
    return "yes" if (not pd.isna(v) and float(v) == 1) else "no"


def build_prompt(row, label=None) -> str:
    """Describe a patient in one sentence from raw (unscaled) clinical values.

    LVEF is stored as a fraction (0.45), so it is converted to percent; an
    earlier version cast it straight to int and wrote "LVEF: 0 percent" for
    every patient. Missing age or LVEF is left out rather than invented.
    `label` is appended only when explicitly given, so inference prompts
    cannot leak the answer.
    """
    age = row.get("Age", np.nan)
    who = f"a {int(age)}-year-old" if not pd.isna(age) else "an adult"
    text = (f"A coronary angiography of {who} {str(row.get('Sex', '')).lower()} patient. "
            f"Diabetes: {_yes_no(row.get('DM'))}. Hypertension: {_yes_no(row.get('HTN'))}. "
            f"Smoker: {_yes_no(row.get('Current Smoker'))}. "
            f"Hyperlipidemia: {_yes_no(row.get('Hyperlipidemia'))}.")
    lvef = row.get("LVEF", np.nan)
    if not pd.isna(lvef) and float(lvef) > 0:
        lvef = float(lvef) * 100 if float(lvef) <= 1 else float(lvef)
        text += f" LVEF: {int(round(lvef))} percent."
    if label is not None:
        text += f" The finding is {'significant stenosis' if label == 1 else 'normal coronary arteries'}."
    return " ".join(text.split())
