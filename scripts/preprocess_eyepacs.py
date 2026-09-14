"""One-time resize of EyePACS's native-resolution images to 512x512,
matching ODIR-5K's own `preprocessed_images` convention exactly (confirmed
by inspection: ODIR-5K's shipped preprocessed files are all precisely
512x512, a forced non-aspect-preserving resize, not aspect-preserving).

Why this exists: quality.py's FOV-detection/circularity metrics scale with
pixel count and are not pre-resized internally (unlike the blur metric,
which already downsamples to BLUR_RESIZE=512 for exactly this reason).
Benchmarked directly on real EyePACS images before deciding to write this:
compute_metrics() alone averaged ~545ms/image at native resolution (up to
5184x3456), which would make quality.py's single pass over all 35,126
images take on the order of 8 hours -- not a "let it run" cost, a design
mismatch: this pipeline's quality/dedupe steps were tuned against
already-512x512 ODIR-5K images and were never exercised against a
dataset shipping native resolution.

Resizing once, up front, to the same 512x512 ODIR-5K's own images already
sit at makes quality.py and dedupe.py's reject-rate/threshold-transfer
comparison genuinely apples-to-apples (same resolution both datasets are
actually scored at), not just faster -- a methodology fix, not only a
performance one.

Deliberately CPU-only multiprocessing (CLAUDE.md: everything except
`train` stays CPU-first) -- GPU JPEG decode/resize would need a materially
different pipeline (nvJPEG/DALI) for a one-time cost multiprocessing
across 8 cores already reduces from ~82 minutes to ~15.

Run: python scripts/preprocess_eyepacs.py
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SRC_DIR = Path("D:/retinaprep_data/eyepacs/train")
DEST_DIR = Path("D:/retinaprep_data/eyepacs/preprocessed_512")
TARGET_SIZE = (512, 512)


def _resize_one(name: str) -> str | None:
    src = SRC_DIR / name
    dest = DEST_DIR / name
    if dest.exists():
        return None
    try:
        with Image.open(src) as im:
            resized = im.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS)
            resized.save(dest, quality=90)
        return None
    except Exception as exc:  # noqa: BLE001 -- log and continue, never drop silently
        return f"{name}: {exc}"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(REPO_ROOT / "artifacts" / "eyepacs" / "manifest.parquet")
    names = [Path(p).name for p in manifest["image_path"]]

    print(f"Resizing {len(names)} images to {TARGET_SIZE} using 8 processes...")
    t0 = time.time()
    errors = []
    with Pool(8) as pool:
        for i, result in enumerate(pool.imap_unordered(_resize_one, names, chunksize=64)):
            if result is not None:
                errors.append(result)
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(names)} done ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. {len(errors)} error(s).")
    if errors:
        print("Errors:", errors[:20])


if __name__ == "__main__":
    main()
