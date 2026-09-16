"""DRIVE adapter -- vessel segmentation, 20 train + 20 test.

**Provenance caveat, load-bearing.** DRIVE's official Grand Challenge
distribution withholds its test annotations entirely (WALKTHROUGH.md
§13), so a complete copy cannot be obtained through the authoritative
route. The copy this adapter reads comes from a third-party Kaggle
re-upload that appears to reproduce the pre-Grand-Challenge
distribution. Its structure was verified before use -- dimensions match,
masks are binary and non-empty with plausible vessel fractions, the two
observers differ on every image, and the test set's first-observer
conventions are consistent with the official training set's -- but
provenance beyond those checks is unverified and every DRIVE number in
this project carries that caveat.

The layout is flat (the Kaggle API does not preserve directories), which
is harmless because the filenames are unambiguous: images 01-20 are the
test split and carry `_manual1` *and* `_manual2`; images 21-40 are the
training split and carry `_manual1` only.

`observer` selects which annotator populates `mask_path`. Both are
available for the test split, and scoring a model against each in turn
is what distinguishes "learned vessels" from "learned observer 1's
style" -- see WALKTHROUGH.md §15.

Two nulls, for the same reason as the other segmentation sets: DRIVE
ships no laterality and no per-image diagnosis, so `eye` and `label` are
genuinely unknown. And as with IDRiD, one image per subject makes
patient-level grouping vacuous here; CHASE_DB1 is the only segmentation
dataset in this project where it does real work.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from retinaprep.adapters.base import register
from retinaprep.config import resolve_path
from retinaprep.utils import MANIFEST_COLUMNS, get_logger

logger = get_logger(__name__)

OBSERVERS = {"1st": "manual1", "2nd": "manual2"}
TEST_IDS = [f"{i:02d}" for i in range(1, 21)]
TRAIN_IDS = [f"{i:02d}" for i in range(21, 41)]


@register("drive")
def build_manifest(cfg: dict) -> pd.DataFrame:
    dataset_cfg = cfg["dataset"]
    root = resolve_path(cfg, cfg["paths"]["data_root"])
    observer = dataset_cfg.get("observer", "1st")
    if observer not in OBSERVERS:
        raise ValueError(f"Unknown observer {observer!r}; expected one of {sorted(OBSERVERS)}")
    suffix = OBSERVERS[observer]

    rows = []
    for split, ids, image_suffix in (
        ("train", TRAIN_IDS, "training"),
        ("test", TEST_IDS, "test"),
    ):
        for sid in ids:
            image_path = root / f"{sid}_{image_suffix}.tif"
            mask_path = root / f"{sid}_{suffix}.gif"
            if not image_path.exists():
                raise FileNotFoundError(f"{image_path} missing -- incomplete DRIVE copy")
            if not mask_path.exists():
                # the training split legitimately has no second observer
                if split == "train" and observer == "2nd":
                    continue
                raise FileNotFoundError(
                    f"{mask_path} missing -- this copy lacks {observer} observer "
                    f"annotations for the {split} split, which is exactly the "
                    "truncation documented in WALKTHROUGH.md section 13"
                )
            rows.append(
                {
                    "image_path": str(image_path),
                    "patient_id": sid,
                    "eye": None,
                    "age": np.nan,
                    "sex": None,
                    "label": None,
                    "dataset_name": dataset_cfg["name"],
                    "mask_path": str(mask_path),
                    "source_split": split,
                    "observer": observer,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No DRIVE image/mask pairs found under {root}")
    logger.info(
        "DRIVE (%s observer): %d pairs (train=%d, test=%d)",
        observer, len(df),
        (df["source_split"] == "train").sum(), (df["source_split"] == "test").sum(),
    )
    return df[[*MANIFEST_COLUMNS, "mask_path", "source_split", "observer"]].reset_index(drop=True)
