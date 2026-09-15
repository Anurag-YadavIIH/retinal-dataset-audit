"""IDRiD segmentation adapter (the "A. Segmentation" release).

IEEE DataPort, CC BY 4.0. 81 images -- 54 training, 27 testing -- each
with pixel-level masks for up to five classes: optic disc plus four
lesion types (microaneurysms, haemorrhages, hard exudates, soft
exudates).

Two things about this release shape the adapter, and both are stated
here rather than rediscovered downstream:

**Not every class exists for every image, and an absent lesion is an
absent FILE, not an empty mask.** Soft exudates are present for only
26/54 training and 14/27 testing images, because most eyes simply do not
have them. A row with no mask for the requested class is dropped with a
count logged -- it is not a defect, and mask QC must not flag it as one.
Optic disc, by contrast, is present for all 81, which is why it is the
default class and the segmentation baseline's target.

**`eye` and `label` are genuinely unknown here.** Filenames are
`IDRiD_01.jpg`: no laterality, and the segmentation release ships no DR
grade (grading is a separate download with its own image set). Both are
emitted as null, which `validate_manifest(task="segmentation")` permits
precisely because inventing them would be worse. One consequence worth
naming: with one image per patient, patient-level splitting on IDRiD is
*identical* to image-level splitting -- this project's central discipline
is vacuous here, and only CHASE_DB1 among the segmentation datasets makes
it non-vacuous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retinaprep.adapters.base import register
from retinaprep.config import resolve_path
from retinaprep.utils import MANIFEST_COLUMNS, get_logger

logger = get_logger(__name__)

# Directory name -> filename suffix, as shipped.
MASK_CLASSES = {
    "OD": ("5. Optic Disc", "_OD"),
    "MA": ("1. Microaneurysms", "_MA"),
    "HE": ("2. Haemorrhages", "_HE"),
    "EX": ("3. Hard Exudates", "_EX"),
    "SE": ("4. Soft Exudates", "_SE"),
}
SPLIT_DIRS = {"train": "a. Training Set", "test": "b. Testing Set"}


@register("idrid_seg")
def build_manifest(cfg: dict) -> pd.DataFrame:
    """Build a segmentation manifest for one IDRiD mask class."""
    dataset_cfg = cfg["dataset"]
    data_root = cfg["paths"]["data_root"]
    root = resolve_path(cfg, data_root)
    mask_class = dataset_cfg.get("mask_class", "OD")
    if mask_class not in MASK_CLASSES:
        raise ValueError(
            f"Unknown mask_class {mask_class!r}; expected one of {sorted(MASK_CLASSES)}"
        )
    mask_dir_name, suffix = MASK_CLASSES[mask_class]

    rows = []
    missing_by_split = {}
    for split, split_dir in SPLIT_DIRS.items():
        image_dir = root / "1. Original Images" / split_dir
        mask_dir = root / "2. All Segmentation Groundtruths" / split_dir / mask_dir_name
        if not image_dir.is_dir():
            raise FileNotFoundError(f"{image_dir} does not exist -- check paths.data_root")

        n_missing = 0
        for image_path in sorted(image_dir.glob("*.jpg")):
            stem = image_path.stem
            mask_path = mask_dir / f"{stem}{suffix}.tif"
            if not mask_path.exists():
                n_missing += 1
                continue
            rows.append(
                {
                    "image_path": str(image_path),
                    "patient_id": stem,
                    "eye": None,
                    "age": np.nan,
                    "sex": None,
                    "label": None,
                    "dataset_name": dataset_cfg["name"],
                    "mask_path": str(mask_path),
                    "source_split": split,
                }
            )
        missing_by_split[split] = n_missing

    df = pd.DataFrame(rows)
    total_missing = sum(missing_by_split.values())
    logger.info(
        "IDRiD %s: %d image/mask pairs (train=%d, test=%d); %d image(s) have no %s mask "
        "(absent lesion, not a defect): %s",
        mask_class,
        len(df),
        (df["source_split"] == "train").sum(),
        (df["source_split"] == "test").sum(),
        total_missing,
        mask_class,
        missing_by_split,
    )
    if df.empty:
        raise ValueError(f"No image/mask pairs found for class {mask_class!r} under {root}")

    return df[[*MANIFEST_COLUMNS, "mask_path", "source_split"]].reset_index(drop=True)


def available_mask_classes(root: Path) -> dict[str, int]:
    """Count image/mask pairs per class -- used by mask QC to report class
    coverage without re-deriving the layout."""
    counts = {}
    for cls, (mask_dir_name, suffix) in MASK_CLASSES.items():
        n = 0
        for split_dir in SPLIT_DIRS.values():
            mask_dir = root / "2. All Segmentation Groundtruths" / split_dir / mask_dir_name
            if mask_dir.is_dir():
                n += len(list(mask_dir.glob(f"*{suffix}.tif")))
        counts[cls] = n
    return counts
