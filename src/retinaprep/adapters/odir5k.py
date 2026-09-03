"""ODIR-5K adapter.

Kaggle: andrewmvd/ocular-disease-recognition-odir5k

Why this dataset: it is organised by patient, with a left and a right fundus
image for each. That structure is exactly what makes patient-level leakage
measurable, and it is why a naive image-level split is so badly wrong here —
bilateral disease means the two eyes of one patient are correlated, so the
"unseen" test image may be the fellow eye of a training image.

The shipped `full_df.csv` has roughly these columns:
    ID, Patient Age, Patient Sex,
    Left-Fundus, Right-Fundus,
    Left-Diagnostic Keywords, Right-Diagnostic Keywords,
    N, D, G, C, A, H, M, O,
    filepath, labels, target, filename

Note the one-hot N/D/G/.../O columns are PATIENT-level, not eye-level. Mapping
them onto individual eyes is a real modelling decision, not a formality.
Document whichever rule is chosen in the README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retinaprep.adapters.base import register
from retinaprep.config import resolve_path
from retinaprep.utils import MANIFEST_COLUMNS, get_logger

logger = get_logger(__name__)

_SIDES = [
    ("L", "Left-Fundus", "Left-Diagnostic Keywords"),
    ("R", "Right-Fundus", "Right-Diagnostic Keywords"),
]


@register("odir5k")
def build_manifest(cfg: dict) -> pd.DataFrame:
    """Build the canonical manifest from ODIR-5K's wide, per-patient CSV.

    See the module docstring for the assumed raw-CSV column names — they are
    unverified against the real download as of writing (see README/session
    notes for what to double check once `full_df.csv` lands).
    """
    dataset_cfg = cfg["dataset"]
    data_root = cfg["paths"]["data_root"]
    metadata_path = resolve_path(cfg, data_root, dataset_cfg["metadata_csv"])
    image_dir = resolve_path(cfg, data_root, dataset_cfg["image_dir"])

    raw = pd.read_csv(metadata_path)

    melted = _melt_eyes(raw)
    melted = _assign_labels(melted, cfg["label"])
    melted = _resolve_and_filter_paths(melted, image_dir)
    melted = _subsample_by_patient(melted, dataset_cfg.get("subsample_n"), cfg["seed"])

    melted["dataset_name"] = dataset_cfg["name"]
    melted["age"] = melted["Patient Age"].astype(float)
    melted["sex"] = melted["Patient Sex"]

    manifest = melted[MANIFEST_COLUMNS].reset_index(drop=True)
    return manifest


def _melt_eyes(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per patient -> one row per eye.

    Age, sex and the patient-level N flag are duplicated onto both eyes;
    the per-eye fundus filename and diagnostic keywords are not.
    """
    parts = []
    for eye, fundus_col, kw_col in _SIDES:
        part = raw[["ID", "Patient Age", "Patient Sex", "N"]].copy()
        part["eye"] = eye
        part["_filename"] = raw[fundus_col]
        part["_keywords"] = raw[kw_col]
        parts.append(part)
    melted = pd.concat(parts, ignore_index=True)
    melted["patient_id"] = melted["ID"].astype(str)
    return melted


def _assign_labels(melted: pd.DataFrame, label_cfg: dict) -> pd.DataFrame:
    normal_col = label_cfg.get("normal_column", "N")

    missing_flag = melted[normal_col].isna().sum()
    if missing_flag:
        logger.warning(
            "%d rows have a missing %r flag; treated as abnormal (label=1) by default",
            missing_flag,
            normal_col,
        )
    label_from_normal_column = melted[normal_col].apply(lambda v: 0 if v == 1 else 1)

    missing_kw = melted["_keywords"].isna().sum()
    if missing_kw:
        logger.warning(
            "%d rows have missing diagnostic keywords; treated as abnormal (label=1) by default",
            missing_kw,
        )
    label_from_keywords = melted["_keywords"].apply(_label_from_keyword_string)

    agree = label_from_normal_column == label_from_keywords
    logger.info(
        "Label rule 'normal_column': %d/%d rows normal (0)",
        int((label_from_normal_column == 0).sum()),
        len(melted),
    )
    logger.info(
        "Label rule 'keywords': %d/%d rows normal (0)",
        int((label_from_keywords == 0).sum()),
        len(melted),
    )
    logger.info(
        "Disagreement between the two label rules: %.1f%% of rows — document this in the README",
        float((~agree).mean()) * 100,
    )

    strategy = label_cfg["strategy"]
    if strategy == "normal_column":
        melted["label"] = label_from_normal_column
    elif strategy == "keywords":
        melted["label"] = label_from_keywords
    else:
        raise ValueError(
            f"Unknown label.strategy {strategy!r}, expected 'normal_column' or 'keywords'"
        )
    melted["label"] = melted["label"].astype(int)
    return melted


def _label_from_keyword_string(keywords: object) -> int:
    if not isinstance(keywords, str) or not keywords.strip():
        return 1
    return 0 if "normal fundus" in keywords.lower() else 1


def _resolve_and_filter_paths(melted: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    melted = melted.copy()
    melted["image_path"] = melted["_filename"].apply(lambda name: str(image_dir / str(name)))
    exists_mask = melted["image_path"].apply(lambda p: Path(p).exists())

    for _, row in melted.loc[~exists_mask].iterrows():
        logger.warning(
            "Dropping patient %s eye %s: image not found at %s",
            row["patient_id"],
            row["eye"],
            row["image_path"],
        )

    n_dropped = int((~exists_mask).sum())
    if n_dropped:
        logger.warning("Dropped %d/%d rows for missing image files", n_dropped, len(melted))

    return melted.loc[exists_mask].reset_index(drop=True)


def _subsample_by_patient(melted: pd.DataFrame, subsample_n: int | None, seed: int) -> pd.DataFrame:
    """Cap the manifest to roughly `subsample_n` rows by keeping whole patients.

    Subsampling by image would silently split a patient's two eyes across the
    subsample boundary and undermine the whole leakage experiment, so patients
    are drawn whole and accumulated until the row-count target is reached (the
    last patient added may push the total slightly over subsample_n).
    """
    if subsample_n is None or len(melted) <= subsample_n:
        return melted

    patient_sizes = melted.groupby("patient_id").size()
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
    return melted.loc[melted["patient_id"].isin(kept_patients)].reset_index(drop=True)
