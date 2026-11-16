"""Old import path, kept so that `from src.angiocad import ...` still works.

This file used to hold its own copies of the datasets, the loss and the
models. Those copies had drifted away from the ones in the notebooks (other
dropout rates, BatchNorm instead of LayerNorm, an image dataset that averaged
the pixels of all frames of a patient), which is why checkpoints trained in
one place would not load in another. Everything now lives in
`angiocad_common`; this module only re-exports the names.
"""
from __future__ import annotations

import warnings

try:  # imported as part of the package: `from src.angiocad import ...`
    from .angiocad_common import *  # noqa: F401,F403
    from .angiocad_common import (ClinicalDataset, ClinicalOnlyModel, FocalLoss,
                                  FrameDataset, FusionModel, ImageOnlyModel,
                                  get_transforms, load_data)
except ImportError:  # imported with src/ on the path: `import angiocad`
    from angiocad_common import *  # noqa: F401,F403
    from angiocad_common import (ClinicalDataset, ClinicalOnlyModel, FocalLoss,
                                 FrameDataset, FusionModel, ImageOnlyModel,
                                 get_transforms, load_data)

warnings.warn("src.angiocad is deprecated; import from angiocad_common instead.",
              DeprecationWarning, stacklevel=2)

train_tfms = get_transforms(train=True)
eval_tfms = get_transforms(train=False)
