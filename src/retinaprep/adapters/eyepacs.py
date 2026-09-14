"""EyePACS / Kaggle Diabetic Retinopathy Detection adapter.

Kaggle: c/diabetic-retinopathy-detection (competition, not a plain dataset --
downloading requires accepting the competition rules on kaggle.com once per
account before the API will serve files, confirmed the hard way this session).

Train split only (35,126 images, confirmed via `trainLabels.csv`: 35126 data
rows). The full competition also ships a labelled test split (53,576 more
images, ~50GB more), but every check this adapter exists for -- patient-level
leakage, fellow-eye concordance, dedupe-threshold transfer, quality scoring,
site recoverability -- only needs one labelled pool with patient IDs, not a
train/test comparison. Restricting to train cuts the download from ~82GB to
~32.6GB for zero loss of anything this project measures.

`trainLabels.csv` columns confirmed against the real downloaded file:
    image, level
    10_left, 0
    10_right, 0

`image` has no file extension (confirmed: real files are `<image>.jpeg`, one
row per eye, no wide/long ambiguity like ODIR-5K's csv had -- this format is
already long). `image` itself encodes both patient_id and eye
(`<patient_id>_<left|right>`), unlike ODIR-5K where eye came from a
same-purpose but differently-shaped filename column; parsed with the same
fail-loudly discipline as odir5k.py's `_derive_eye_column` rather than
trusting the split blindly.

`level` is the standard ETDRS-referenced 0-4 DR severity grade. Binary label
= 1 for level >= 2 (moderate-or-worse NPDR), the standard clinical
"referable DR" threshold used throughout the DR-screening literature -- not
configurable, the same way odir5k.py's "normal fundus" keyword match isn't;
both are fixed clinical/textual definitions, not tunable thresholds.

No age/sex metadata is published with this dataset (unlike ODIR-5K) --
both columns are left null, which the canonical manifest contract already
allows (`utils.REQUIRED_NON_NULL` doesn't include them).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from retinaprep.adapters.base import register, subsample_by_patient
from retinaprep.config import resolve_path
from retinaprep.utils import MANIFEST_COLUMNS, get_logger

logger = get_logger(__name__)

_IMAGE_ID_PATTERN = re.compile(r"^(?P<patient_id>\d+)_(?P<eye>left|right)$", re.IGNORECASE)
REFERABLE_DR_THRESHOLD = 2


@register("eyepacs")
def build_manifest(cfg: dict) -> pd.DataFrame:
    """Build the canonical manifest from EyePACS's train split."""
    dataset_cfg = cfg["dataset"]
    data_root = cfg["paths"]["data_root"]
    metadata_path = resolve_path(cfg, data_root, dataset_cfg["metadata_csv"])
    image_dir = resolve_path(cfg, data_root, dataset_cfg["image_dir"])
    image_ext = dataset_cfg.get("image_ext", "jpeg")

    df = pd.read_csv(metadata_path)
    df["patient_id"], df["eye"] = _parse_image_id(df["image"])
    _report_eyes_per_patient(df)

    df["label"] = (df["level"] >= REFERABLE_DR_THRESHOLD).astype(int)
    logger.info(
        "Label rule 'level >= %d' (referable DR): %d/%d rows normal (0)",
        REFERABLE_DR_THRESHOLD,
        int((df["label"] == 0).sum()),
        len(df),
    )

    df["image_path"] = df["image"].apply(lambda name: str(image_dir / f"{name}.{image_ext}"))
    df = _filter_to_existing_paths(df)
    df = subsample_by_patient(df, dataset_cfg.get("subsample_n"), cfg["seed"])

    df["dataset_name"] = dataset_cfg["name"]
    df["age"] = np.nan
    df["sex"] = None

    manifest = df[MANIFEST_COLUMNS].reset_index(drop=True)
    return manifest


def _parse_image_id(image_ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    """patient_id and eye ('L'/'R') from `<patient_id>_<left|right>`.

    Every value must match, or fail loudly rather than silently mis-parsing
    an unexpected id -- same discipline as odir5k.py's `_derive_eye_column`.
    """
    parsed = image_ids.astype(str).str.extract(_IMAGE_ID_PATTERN)
    bad_mask = parsed["patient_id"].isna()
    if bad_mask.any():
        bad = image_ids[bad_mask].tolist()
        raise ValueError(
            f"{len(bad)} image id(s) do not match the expected "
            f"'<patient_id>_left'/'<patient_id>_right' pattern: {bad[:10]}"
        )
    eye = parsed["eye"].str.lower().map({"left": "L", "right": "R"})
    return parsed["patient_id"], eye


def _report_eyes_per_patient(df: pd.DataFrame) -> None:
    """Log the single-eye-vs-two-eye patient split -- never assume it's always two."""
    row_counts = df.groupby("patient_id").size()
    n_single = int((row_counts == 1).sum())
    n_double = int((row_counts == 2).sum())
    n_other = int((~row_counts.isin([1, 2])).sum())
    logger.info(
        "Patients with 1 eye: %d, with 2 eyes: %d%s",
        n_single,
        n_double,
        f", with other row counts: {n_other}" if n_other else "",
    )


def _filter_to_existing_paths(df: pd.DataFrame) -> pd.DataFrame:
    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())
    n_dropped = int((~exists_mask).sum())
    if n_dropped:
        for _, row in df.loc[~exists_mask].head(20).iterrows():
            logger.warning(
                "Dropping patient %s eye %s: image not found at %s",
                row["patient_id"],
                row["eye"],
                row["image_path"],
            )
        logger.warning("Dropped %d/%d rows for missing image files", n_dropped, len(df))
    return df.loc[exists_mask].reset_index(drop=True)
