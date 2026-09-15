"""CHASE_DB1 adapter -- vessel segmentation, and the only two-observer set
this project could actually obtain.

28 images from the Child Heart and Health Study in England, each with
vessel masks from **two independent human observers** (`1stHO`, `2ndHO`).
That redundancy is the whole reason this dataset is here: DRIVE's
second-observer annotations are withheld by its official distribution and
absent from every curated mirror checked, and REFUGE merges its seven
graders into a single reference before release (WALKTHROUGH.md §13). The
inter-grader ceiling is computable here and nowhere else among the
candidates.

**It also earns its place for a second, independent reason.** Filenames
are `Image_01L` / `Image_01R` -- left and right eye of the same child --
so 28 images come from **14 subjects**, and `patient_id` is genuinely
recoverable. That makes CHASE_DB1 the only segmentation dataset here
where this project's patient-level splitting discipline is non-vacuous.
On IDRiD (and on REFUGE and RIM-ONE) there is one image per patient, so
grouping by patient and splitting by image are the same operation, and
the discipline demonstrates nothing.

Two caveats carried into every downstream number:

- **`label` is null.** CHASE_DB1 ships no disease label, so
  classification-shaped validation does not apply; this is a
  segmentation manifest.
- **14 subjects is not a sample size.** A patient-grouped split leaves
  roughly 10 subjects to train on and 4 to test. That demonstrates the
  discipline; it does not measure anything, and the write-up says so
  before reporting results rather than after.

`observer` selects which annotator's mask populates `mask_path`
(default `1stHO`). Inter-grader work reads both directly rather than
through the manifest, since the manifest is one-mask-per-row by design.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from retinaprep.adapters.base import register
from retinaprep.config import resolve_path
from retinaprep.utils import MANIFEST_COLUMNS, get_logger

logger = get_logger(__name__)

OBSERVERS = ("1stHO", "2ndHO")
_NAME = re.compile(r"^Image_(?P<subject>\d+)(?P<eye>[LR])$", re.IGNORECASE)


@register("chasedb1")
def build_manifest(cfg: dict) -> pd.DataFrame:
    """Build a segmentation manifest from CHASE_DB1's images and one
    observer's vessel masks."""
    dataset_cfg = cfg["dataset"]
    root = resolve_path(cfg, cfg["paths"]["data_root"])
    observer = dataset_cfg.get("observer", "1stHO")
    if observer not in OBSERVERS:
        raise ValueError(f"Unknown observer {observer!r}; expected one of {OBSERVERS}")

    image_dir = root / dataset_cfg.get("image_dir", "Images")
    mask_dir = root / dataset_cfg.get("mask_dir", "Masks")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{image_dir} does not exist -- check paths.data_root")

    rows = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        stem = image_path.stem
        m = _NAME.match(stem)
        if m is None:
            raise ValueError(
                f"{image_path.name!r} does not match the expected 'Image_<subject><L|R>' "
                "pattern -- patient_id and eye are derived from it, so a silent "
                "mismatch would break patient-level grouping"
            )
        mask_path = mask_dir / f"{stem}_{observer}.png"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"{mask_path} missing -- CHASE_DB1 should carry both observers for "
                "every image; an incomplete copy would silently weaken the "
                "inter-grader comparison this dataset was chosen for"
            )
        rows.append(
            {
                "image_path": str(image_path),
                "patient_id": m.group("subject"),
                "eye": m.group("eye").upper(),
                "age": np.nan,
                "sex": None,
                "label": None,
                "dataset_name": dataset_cfg["name"],
                "mask_path": str(mask_path),
                "observer": observer,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No images found under {image_dir}")

    per_subject = df.groupby("patient_id").size()
    logger.info(
        "CHASE_DB1 (%s): %d images from %d subjects; eyes per subject: min=%d max=%d",
        observer,
        len(df),
        df["patient_id"].nunique(),
        per_subject.min(),
        per_subject.max(),
    )
    return df[[*MANIFEST_COLUMNS, "mask_path", "observer"]].reset_index(drop=True)


def observer_mask_pairs(cfg: dict) -> pd.DataFrame:
    """One row per image with BOTH observers' mask paths -- the input to
    inter-grader agreement, which the one-mask-per-row manifest cannot
    express."""
    first = build_manifest({**cfg, "dataset": {**cfg["dataset"], "observer": "1stHO"}})
    second = build_manifest({**cfg, "dataset": {**cfg["dataset"], "observer": "2ndHO"}})
    merged = first.merge(
        second[["image_path", "mask_path"]],
        on="image_path",
        suffixes=("_1stHO", "_2ndHO"),
    )
    if len(merged) != len(first):
        raise ValueError(f"observer join dropped rows: {len(first)} -> {len(merged)}")
    return merged
