"""Step 1 — build the canonical manifest.

Reads the raw dataset via an adapter, validates that every image opens, logs
anything corrupt, and writes artifacts/manifest.parquet.
"""

from __future__ import annotations

import pandas as pd
from PIL import Image, UnidentifiedImageError

from retinaprep.adapters import get_adapter
from retinaprep.config import resolve_path
from retinaprep.utils import get_logger, set_global_seed, validate_manifest

logger = get_logger(__name__)


def run_ingest(cfg: dict) -> None:
    """Dispatch to the configured adapter, validate, drop corrupt images, persist."""
    set_global_seed(cfg["seed"])

    adapter = get_adapter(cfg["dataset"]["name"])
    manifest = adapter(cfg)
    # Segmentation adapters satisfy a different non-null subset of the same
    # schema (utils.SEGMENTATION_REQUIRED_NON_NULL); defaults to
    # classification so every existing config behaves identically.
    validate_manifest(manifest, task=cfg["dataset"].get("task", "classification"))

    manifest, corrupt = _drop_unopenable_images(manifest)

    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if corrupt:
        corrupt_path = artifacts_dir / "corrupt_images.csv"
        pd.DataFrame(corrupt).to_csv(corrupt_path, index=False)
        logger.warning("Logged %d corrupt/unreadable images to %s", len(corrupt), corrupt_path)

    manifest_path = artifacts_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logger.info("Wrote manifest (%d rows) to %s", len(manifest), manifest_path)

    _print_summary(manifest)


def _drop_unopenable_images(manifest: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Force a full decode of every image; bad files are logged, not silently dropped.

    Masks are decoded too when present: an unreadable mask makes the row
    useless for segmentation just as surely as an unreadable image, and
    finding out at ingest is cheaper than mid-training.
    """
    corrupt: list[dict] = []
    keep: list[bool] = []
    has_masks = "mask_path" in manifest.columns
    for row in manifest.itertuples(index=False):
        image_path = row.image_path
        paths = [image_path]
        if has_masks and isinstance(getattr(row, "mask_path", None), str):
            paths.append(row.mask_path)
        try:
            for p in paths:
                with Image.open(p) as im:
                    im.load()
            keep.append(True)
        except (OSError, UnidentifiedImageError) as exc:
            logger.warning("Corrupt or unreadable file for %s: %s", image_path, exc)
            corrupt.append({"image_path": image_path, "reason": str(exc)})
            keep.append(False)
    clean = manifest.loc[keep].reset_index(drop=True)
    return clean, corrupt


def _print_summary(manifest: pd.DataFrame) -> None:
    n_images = len(manifest)
    n_patients = manifest["patient_id"].nunique()
    class_balance = manifest["label"].value_counts(normalize=True).sort_index().round(3).to_dict()
    images_per_patient = manifest.groupby("patient_id").size()

    print(f"Manifest summary: {n_images} images, {n_patients} patients")
    print(f"Class balance (label -> fraction): {class_balance}")
    print(
        "Images per patient: "
        f"mean={images_per_patient.mean():.2f} "
        f"min={images_per_patient.min()} "
        f"max={images_per_patient.max()}"
    )
