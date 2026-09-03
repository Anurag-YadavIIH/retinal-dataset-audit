"""Shared helpers: seeding, logging, run bookkeeping."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    """Seed random, numpy and torch.

    Determinism is not a nicety here — the whole project is a comparison
    between runs, so an unseeded run makes the headline number meaningless.

    torch is only a hard dependency of `train`; every other module (this
    includes ingest and split, which use the seed for patient subsampling and
    the stratified-group split) must run on a CPU-only environment that may
    not have torch installed. Seeding it is best-effort and logged, not
    silently skipped.
    """
    logger = get_logger(__name__)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.warning(
            "torch is not installed; skipped torch seeding (only load-bearing for `train`)"
        )

    logger.info("Global seed set to %d", seed)


def get_logger(name: str) -> logging.Logger:
    """Module logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


MANIFEST_COLUMNS = [
    "image_path",
    "patient_id",
    "eye",
    "age",
    "sex",
    "label",
    "dataset_name",
]


REQUIRED_NON_NULL = ["image_path", "patient_id", "eye", "label", "dataset_name"]


def validate_manifest(df: pd.DataFrame) -> None:
    """Assert the canonical manifest contract.

    Collects every violation before raising, rather than failing on the
    first one, so a caller fixing an adapter sees the whole picture in one
    run instead of playing whack-a-mole.
    """
    missing_cols = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {missing_cols}")

    errors: list[str] = []

    for col in REQUIRED_NON_NULL:
        bad_rows = df.index[df[col].isna()].tolist()
        if bad_rows:
            errors.append(f"column {col!r} is null at rows {bad_rows}")

    bad_eye_mask = df["eye"].notna() & ~df["eye"].isin(["L", "R"])
    if bad_eye_mask.any():
        bad_rows = df.index[bad_eye_mask].tolist()
        bad_values = df.loc[bad_eye_mask, "eye"].tolist()
        errors.append(
            f"column 'eye' has values outside {{'L', 'R'}} at rows {bad_rows}: {bad_values}"
        )

    bad_label_mask = df["label"].notna() & ~df["label"].isin([0, 1])
    if bad_label_mask.any():
        bad_rows = df.index[bad_label_mask].tolist()
        bad_values = df.loc[bad_label_mask, "label"].tolist()
        errors.append(
            f"column 'label' has values outside {{0, 1}} at rows {bad_rows}: {bad_values}"
        )

    is_str = df["patient_id"].apply(lambda x: isinstance(x, str))
    non_str_patient_mask = df["patient_id"].notna() & ~is_str
    if non_str_patient_mask.any():
        bad_rows = df.index[non_str_patient_mask].tolist()
        errors.append(f"column 'patient_id' is not a string at rows {bad_rows}")

    non_null_paths = df["image_path"].notna()
    exists = pd.Series(False, index=df.index)
    exists[non_null_paths] = df.loc[non_null_paths, "image_path"].apply(lambda p: Path(p).exists())
    missing_files_mask = non_null_paths & ~exists
    if missing_files_mask.any():
        bad_rows = df.index[missing_files_mask].tolist()
        errors.append(
            f"column 'image_path' points to a nonexistent file at rows {bad_rows}: "
            f"{df.loc[missing_files_mask, 'image_path'].tolist()}"
        )

    if errors:
        raise ValueError("Manifest failed validation:\n" + "\n".join(f"  - {e}" for e in errors))
