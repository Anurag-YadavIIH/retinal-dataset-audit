"""Item 3 step 2: do this project's quality thresholds transfer to EyePACS?

The naive comparison is badly misleading, and this script exists because
the first attempt at it nearly got reported:

    EyePACS reject rate 16.9%  vs  ODIR-5K reject rate 0.7%   (24x)

That number compares EyePACS images that reached 512x512 through
scripts/preprocess_eyepacs.py's PIL-LANCZOS downscale against ODIR-5K
images that reached 512x512 through whatever preprocessing the Kaggle
release itself shipped. Checked directly on a 150-image sample rather
than assumed:

    laplacian_var median, ODIR-5K as shipped at 512x512 : 232.2
    laplacian_var median, ODIR-5K raw -> LANCZOS 512    :  76.8
    laplacian_var median, EyePACS    raw -> LANCZOS 512 :  40.0

Re-deriving ODIR-5K's own 512x512 copies through the same LANCZOS path
drops its blur metric 3x, with no change whatsoever to the underlying
photographs. Most of the apparent 24x reject-rate gap is therefore a
property of the two datasets' *resize histories*, not their image
quality -- precisely the failure mode variance_of_laplacian's docstring
warns about ("it ranks cameras, not sharpness"), except the confound
enters before this module is ever called, so its internal resize can't
defend against it.

This script makes the comparison fair the only way that actually works:
push ODIR-5K's RAW images through the identical LANCZOS-512 path EyePACS
went through, score both with identical config, and report both reject
rates side by side with the as-shipped number kept for reference rather
than quietly replaced.

Run: python notebooks/quality_threshold_transfer.py
Output: printed report + artifacts/eyepacs_512/quality_threshold_transfer.json
"""

from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.config import load_config  # noqa: E402
from retinaprep.quality import compute_metrics, gradability_score  # noqa: E402
from retinaprep.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

ODIR_RAW_DIR = REPO_ROOT / "data" / "odir5k" / "ODIR-5K" / "ODIR-5K" / "Training Images"
ODIR_ARTIFACTS = REPO_ROOT / "artifacts"
EYEPACS_ARTIFACTS = REPO_ROOT / "artifacts" / "eyepacs_512"
TARGET_SIZE = (512, 512)

_CFG = None


def _init_cfg() -> None:
    global _CFG
    _CFG = load_config()


def _score_odir_raw_via_lanczos(name: str) -> float | None:
    """ODIR-5K raw image -> the exact LANCZOS-512 path EyePACS took -> score."""
    path = ODIR_RAW_DIR / name
    try:
        with Image.open(path) as im:
            resized = im.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
        image = np.array(resized)
        return float(gradability_score(compute_metrics(image, _CFG), _CFG))
    except Exception as exc:  # noqa: BLE001 -- report, never drop silently
        logger.warning("Could not score %s: %s", name, exc)
        return None


def main() -> None:
    cfg = load_config()
    reject_threshold = cfg["quality"]["reject_below_score"]

    odir_manifest = pd.read_parquet(ODIR_ARTIFACTS / "manifest.parquet")
    names = [Path(p).name for p in odir_manifest["image_path"]]
    missing = [n for n in names if not (ODIR_RAW_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest image(s) have no raw counterpart in {ODIR_RAW_DIR} "
            f"(first few: {missing[:5]}) -- the fair comparison needs every raw original."
        )

    logger.info("Re-scoring %d ODIR-5K images through the LANCZOS-512 path...", len(names))
    with Pool(6, initializer=_init_cfg) as pool:
        scores = pool.map(_score_odir_raw_via_lanczos, names, chunksize=32)
    odir_refit = pd.Series([s for s in scores if s is not None])

    odir_shipped = pd.read_parquet(ODIR_ARTIFACTS / "quality.parquet")["score"]
    eyepacs = pd.read_parquet(EYEPACS_ARTIFACTS / "quality.parquet")["score"]

    def _summary(s: pd.Series) -> dict:
        return {
            "n": int(len(s)),
            "median_score": float(s.median()),
            "mean_score": float(s.mean()),
            "reject_rate": float((s < reject_threshold).mean()),
            "n_rejected": int((s < reject_threshold).sum()),
        }

    result = {
        "reject_threshold": float(reject_threshold),
        "odir5k_as_shipped": _summary(odir_shipped),
        "odir5k_raw_via_lanczos512": _summary(odir_refit),
        "eyepacs_raw_via_lanczos512": _summary(eyepacs),
    }

    out_path = EYEPACS_ARTIFACTS / "quality_threshold_transfer.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\nWrote {out_path}\n")
    print(f"{'condition':38s} {'n':>7s} {'median':>8s} {'reject%':>9s}")
    for key in ("odir5k_as_shipped", "odir5k_raw_via_lanczos512", "eyepacs_raw_via_lanczos512"):
        r = result[key]
        print(f"{key:38s} {r['n']:7d} {r['median_score']:8.3f} {r['reject_rate']*100:8.1f}%")
    print(
        "\nThe middle row is the only fair comparator for the bottom row: same"
        "\nresize path, same config, same thresholds. The top row is ODIR-5K's"
        "\npublished number, kept for reference, NOT a like-for-like baseline."
    )


if __name__ == "__main__":
    main()
