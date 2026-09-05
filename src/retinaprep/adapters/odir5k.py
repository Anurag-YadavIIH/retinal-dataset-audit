"""ODIR-5K adapter.

Kaggle: andrewmvd/ocular-disease-recognition-odir5k

Why this dataset: it is organised by patient, with a left and a right fundus
image for each. That structure is exactly what makes patient-level leakage
measurable, and it is why a naive image-level split is so badly wrong here —
bilateral disease means the two eyes of one patient are correlated, so the
"unseen" test image may be the fellow eye of a training image.

`full_df.csv` is LONG, one row per eye/image, confirmed against the real
download: 6392 rows, 3358 unique patient IDs, 6392 unique filenames. It is
NOT the wide one-row-per-patient layout this file originally assumed (that
version melted a nonexistent wide layout and would have silently doubled
every row). 6392 / 3358 ≈ 1.90 -- several hundred patients have only one
usable eye, so this adapter never assumes exactly two images per patient.

Columns actually present:
    ID, Patient Age, Patient Sex,
    Left-Fundus, Right-Fundus,
    Left-Diagnostic Keywords, Right-Diagnostic Keywords,
    N, D, G, C, A, H, M, O,
    filepath, labels, target, filename

`Left-Fundus`/`Right-Fundus` are patient-descriptive metadata, not proof
either file exists on disk -- both are populated even for a single-eye
patient. `eye` and `image_path` are derived from `filename` (this row's own
image) instead, which is unambiguous and doesn't depend on that quirk.
`filepath` points at a relative path under the original Kaggle input
mount and is not resolvable from this repo, so it is ignored entirely.

The one-hot N/D/G/.../O columns, and `target`/`labels`, are populated from
the same source patient record on every row belonging to that patient --
`_report_label_locality` checks this empirically rather than assuming it,
and logs whether they ever actually differ between a patient's two rows.
Mapping a patient-level flag onto individual eyes is a real modelling
decision, not a formality; document whichever label.strategy is chosen,
and this patient-vs-eye-level finding, in the README.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from retinaprep.adapters.base import register
from retinaprep.config import resolve_path
from retinaprep.utils import MANIFEST_COLUMNS, get_logger

logger = get_logger(__name__)

_FILENAME_EYE_PATTERN = re.compile(r"_(left|right)\.[^.]+$", re.IGNORECASE)
_KEYWORD_COLUMN_BY_EYE = {"L": "Left-Diagnostic Keywords", "R": "Right-Diagnostic Keywords"}


@register("odir5k")
def build_manifest(cfg: dict) -> pd.DataFrame:
    """Build the canonical manifest from ODIR-5K's long, one-row-per-eye CSV."""
    dataset_cfg = cfg["dataset"]
    data_root = cfg["paths"]["data_root"]
    metadata_path = resolve_path(cfg, data_root, dataset_cfg["metadata_csv"])
    image_dir = resolve_path(cfg, data_root, dataset_cfg["image_dir"])

    df = pd.read_csv(metadata_path)
    df["patient_id"] = df["ID"].astype(str)
    df["eye"] = _derive_eye_column(df["filename"])

    _report_eyes_per_patient(df)
    _report_label_locality(df)

    df = _assign_labels(df, cfg["label"])
    df = _resolve_and_filter_paths(df, image_dir)
    df = _subsample_by_patient(df, dataset_cfg.get("subsample_n"), cfg["seed"])

    df["dataset_name"] = dataset_cfg["name"]
    df["age"] = df["Patient Age"].astype(float)
    df["sex"] = df["Patient Sex"]

    manifest = df[MANIFEST_COLUMNS].reset_index(drop=True)
    return manifest


def _derive_eye_column(filenames: pd.Series) -> pd.Series:
    """L/R from the filename's own `_left`/`_right` suffix.

    Every filename must match, or fail loudly rather than silently coercing
    an unexpected name to a guessed side.
    """
    side = filenames.astype(str).str.extract(_FILENAME_EYE_PATTERN, expand=False)
    bad_mask = side.isna()
    if bad_mask.any():
        bad = filenames[bad_mask].tolist()
        raise ValueError(
            f"{len(bad)} filename(s) do not match the expected "
            f"'..._left.<ext>' / '..._right.<ext>' pattern: {bad}"
        )
    return side.str.lower().map({"left": "L", "right": "R"})


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


def _report_label_locality(df: pd.DataFrame) -> None:
    """Check empirically whether target/labels vary within a patient's rows.

    N/D/G/.../O are already known to be patient-level (module docstring).
    Whether the derived `target`/`labels` columns are too is worth checking
    rather than assuming, since a wrong assumption here silently overstates
    how much per-eye signal the labels actually carry.
    """
    row_counts = df.groupby("patient_id").size()
    multi_eye_ids = row_counts[row_counts > 1].index
    multi_eye_df = df[df["patient_id"].isin(multi_eye_ids)]

    for col in ("target", "labels"):
        if col not in df.columns:
            continue
        n_unique_per_patient = multi_eye_df.groupby("patient_id")[col].nunique()
        n_varying = int((n_unique_per_patient > 1).sum())
        if n_varying == 0:
            logger.info(
                "Column %r never differs between a patient's eyes (checked %d multi-eye "
                "patients) -- it is PATIENT-level, not eye-level. Document this limitation "
                "in the README rather than treating it as eye-specific signal.",
                col,
                len(n_unique_per_patient),
            )
        else:
            logger.info(
                "Column %r differs between eyes for %d/%d multi-eye patients -- eye-level.",
                col,
                n_varying,
                len(n_unique_per_patient),
            )


def _assign_labels(df: pd.DataFrame, label_cfg: dict) -> pd.DataFrame:
    normal_col = label_cfg.get("normal_column", "N")

    missing_flag = df[normal_col].isna().sum()
    if missing_flag:
        logger.warning(
            "%d rows have a missing %r flag; treated as abnormal (label=1) by default",
            missing_flag,
            normal_col,
        )
    label_from_normal_column = df[normal_col].apply(lambda v: 0 if v == 1 else 1)

    own_keywords = np.where(
        df["eye"] == "L", df["Left-Diagnostic Keywords"], df["Right-Diagnostic Keywords"]
    )
    own_keywords = pd.Series(own_keywords, index=df.index)
    missing_kw = own_keywords.isna().sum()
    if missing_kw:
        logger.warning(
            "%d rows have missing diagnostic keywords for their own eye; treated as "
            "abnormal (label=1) by default",
            missing_kw,
        )
    label_from_keywords = own_keywords.apply(_label_from_keyword_string)

    agree = label_from_normal_column == label_from_keywords
    logger.info(
        "Label rule 'normal_column': %d/%d rows normal (0)",
        int((label_from_normal_column == 0).sum()),
        len(df),
    )
    logger.info(
        "Label rule 'keywords': %d/%d rows normal (0)",
        int((label_from_keywords == 0).sum()),
        len(df),
    )
    logger.info(
        "Disagreement between the two label rules: %.1f%% of rows — document this in the README",
        float((~agree).mean()) * 100,
    )

    strategy = label_cfg["strategy"]
    if strategy == "normal_column":
        df["label"] = label_from_normal_column
    elif strategy == "keywords":
        df["label"] = label_from_keywords
    else:
        raise ValueError(
            f"Unknown label.strategy {strategy!r}, expected 'normal_column' or 'keywords'"
        )
    df["label"] = df["label"].astype(int)
    return df


def _label_from_keyword_string(keywords: object) -> int:
    if not isinstance(keywords, str) or not keywords.strip():
        return 1
    return 0 if "normal fundus" in keywords.lower() else 1


def _resolve_and_filter_paths(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Resolve by `filename` against image_dir. `filepath` is ignored -- it points
    at a relative path under the original Kaggle input mount, not this repo."""
    df = df.copy()
    df["image_path"] = df["filename"].apply(lambda name: str(image_dir / str(name)))
    exists_mask = df["image_path"].apply(lambda p: Path(p).exists())

    for _, row in df.loc[~exists_mask].iterrows():
        logger.warning(
            "Dropping patient %s eye %s: image not found at %s",
            row["patient_id"],
            row["eye"],
            row["image_path"],
        )

    n_dropped = int((~exists_mask).sum())
    if n_dropped:
        logger.warning("Dropped %d/%d rows for missing image files", n_dropped, len(df))

    return df.loc[exists_mask].reset_index(drop=True)


def _subsample_by_patient(df: pd.DataFrame, subsample_n: int | None, seed: int) -> pd.DataFrame:
    """Cap the manifest to roughly `subsample_n` rows by keeping whole patients.

    Subsampling by image would silently split a patient's eyes across the
    subsample boundary and undermine the whole leakage experiment, so patients
    are drawn whole (be it one eye or two) and accumulated until the row-count
    target is reached (the last patient added may push the total slightly
    over subsample_n).
    """
    if subsample_n is None or len(df) <= subsample_n:
        return df

    patient_sizes = df.groupby("patient_id").size()
    rng = np.random.default_rng(seed)
    shuffled_patients = rng.permutation(patient_sizes.index.to_numpy())

    kept_patients = []
    running_total = 0
    for pid in shuffled_patients:
        if running_total >= subsample_n:
            break
        kept_patients.append(pid)
        running_total += int(patient_sizes[pid])

    logger.info(
        "Subsampled to %d whole patients (%d rows), target was %d rows",
        len(kept_patients),
        running_total,
        subsample_n,
    )
    return df.loc[df["patient_id"].isin(kept_patients)].reset_index(drop=True)
