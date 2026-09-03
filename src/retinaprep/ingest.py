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
    validate_manifest(manifest)

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
    """Force a full decode of every image; bad files are logged, not silently dropped."""
    corrupt: list[dict] = []
    keep: list[bool] = []
    for image_path in manifest["image_path"]:
        try:
            with Image.open(image_path) as im:
                im.load()
            keep.append(True)
        except (OSError, UnidentifiedImageError) as exc:
            logger.warning("Corrupt or unreadable image %s: %s", image_path, exc)
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
