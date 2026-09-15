"""Shared helpers: seeding, logging, run bookkeeping."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
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


def config_hash(cfg: dict) -> str:
    """Short, stable hash of the resolved config.

    Excludes `_config_path` (an absolute filesystem path set by
    load_config, not actual config content -- including it would make the
    hash machine-dependent and change on every checkout location). Used to
    tell whether two runs used the same config without diffing full JSON
    blobs, and to group/select run cohorts in artifacts/runs/index.json.
    """
    payload = {k: v for k, v in cfg.items() if k != "_config_path"}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def run_timestamp() -> str:
    """UTC timestamp for a run directory name: sortable, filesystem-safe."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def append_run_index(artifacts_dir: Path, entry: dict) -> None:
    """Append one run's summary to artifacts/runs/index.json.

    artifacts/runs/<dir>/metrics.json has the full detail for one run, but
    each run now lives in its own uniquely-named directory (train.py no
    longer overwrites same-named runs) -- this is the one file that lists
    every run that has ever happened, so a figure can be regenerated from
    disk, and so a stale/superseded cohort is discoverable rather than
    silently gone.
    """
    index_path = artifacts_dir / "runs" / "index.json"
    entries = load_run_index(artifacts_dir)
    entries.append(entry)
    save_json(entries, index_path)


def load_run_index(artifacts_dir: Path) -> list[dict]:
    """Every run ever recorded, oldest first. [] if index.json doesn't exist yet."""
    index_path = artifacts_dir / "runs" / "index.json"
    if not index_path.exists():
        return []
    with open(index_path) as fh:
        return json.load(fh)


def load_current_run_metrics(artifacts_dir: Path) -> pd.DataFrame:
    """One row per run, restricted to each arm's most recent config-hash cohort.

    Runs no longer overwrite, so multiple historical cohorts for the same
    arm (different configs, different points in the project's history) can
    coexist under artifacts/runs/. Aggregating blindly across all of them
    would silently blend incompatible runs into one meaningless average.
    This selects, per arm, only the runs sharing that arm's most recently
    recorded config_hash -- older cohorts stay on disk and in index.json
    for provenance, but drop out of the current headline numbers.
    """
    entries = load_run_index(artifacts_dir)
    if not entries:
        return pd.DataFrame()
    df = pd.DataFrame(entries)
    latest_hash_by_arm = df.sort_values("timestamp").groupby("arm")["config_hash"].last()
    is_current = df["config_hash"] == df["arm"].map(latest_hash_by_arm)
    return df[is_current].reset_index(drop=True)


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

# Segmentation datasets satisfy a different subset of the same schema, and
# pretending otherwise would mean inventing values. IDRiD's segmentation
# release ships no laterality and no DR grade (its grading set is a
# separate download), and CHASE_DB1 has no disease label at all -- so `eye`
# and `label` are genuinely unknown there, not merely unpopulated. What a
# segmentation row must have instead is a mask.
#
# The columns themselves stay identical either way: MANIFEST_COLUMNS is
# unchanged, so every adapter still emits all seven and nothing downstream
# sees a different shape. Only which of them may be null varies by task.
SEGMENTATION_REQUIRED_NON_NULL = ["image_path", "patient_id", "mask_path", "dataset_name"]

# Additive, and deliberately not in MANIFEST_COLUMNS: classification
# adapters neither emit it nor are affected by it.
OPTIONAL_COLUMNS = ["mask_path"]


def validate_manifest(df: pd.DataFrame, task: str = "classification") -> None:
    """Assert the canonical manifest contract.

    Collects every violation before raising, rather than failing on the
    first one, so a caller fixing an adapter sees the whole picture in one
    run instead of playing whack-a-mole.

    `task` selects which columns must be non-null; it does not change the
    required column set. Defaults to "classification" so every existing
    caller keeps exactly the guarantees it had.
    """
    if task not in ("classification", "segmentation"):
        raise ValueError(f"Unknown task {task!r}, expected 'classification' or 'segmentation'")

    missing_cols = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if task == "segmentation" and "mask_path" not in df.columns:
        missing_cols.append("mask_path")
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {missing_cols}")

    errors: list[str] = []
    required_non_null = (
        REQUIRED_NON_NULL if task == "classification" else SEGMENTATION_REQUIRED_NON_NULL
    )

    for col in required_non_null:
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

    if "mask_path" in df.columns:
        non_null_masks = df["mask_path"].notna()
        mask_exists = pd.Series(False, index=df.index)
        mask_exists[non_null_masks] = df.loc[non_null_masks, "mask_path"].apply(
            lambda p: Path(p).exists()
        )
        missing_masks = non_null_masks & ~mask_exists
        if missing_masks.any():
            bad_rows = df.index[missing_masks].tolist()
            errors.append(
                f"column 'mask_path' points to a nonexistent file at rows {bad_rows}: "
                f"{df.loc[missing_masks, 'mask_path'].tolist()}"
            )

    if errors:
        raise ValueError("Manifest failed validation:\n" + "\n".join(f"  - {e}" for e in errors))
