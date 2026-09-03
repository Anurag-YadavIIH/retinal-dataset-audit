"""Fixtures generating synthetic fundus-like images so tests run without the
real dataset. CI must never need a 2GB Kaggle download."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from retinaprep.config import load_config

IMAGE_SIZE = 64


def _fundus_image(rng: np.random.Generator, brightness: int, dim_left: bool = False) -> Image.Image:
    """Bright circle (fundus/FOV) on a black background, plus a little texture."""
    size = IMAGE_SIZE
    arr = np.zeros((size, size, 3), dtype=np.int16)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size // 2
    r = size // 2 - 4
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
    arr[mask] = brightness

    noise = rng.integers(0, 15, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    if dim_left:
        arr[:, : size // 2] = (arr[:, : size // 2] * 0.25).astype(np.int16)

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


@pytest.fixture
def synthetic_fundus_dir(tmp_path):
    """Generate a synthetic ODIR-5K-shaped raw dataset: an image directory plus
    a wide per-patient metadata CSV at the conventional `full_df.csv` path.

    10 base patients (20 images, alternating normal/abnormal) plus 4 patients
    hosting deliberately degraded variants, each paired with one clean
    companion eye to keep the wide left/right layout intact:
      - patient 11: blurred right eye (Gaussian blur)
      - patient 12: darkened left eye (low exposure)
      - patient 13: one-side-dimmed left eye (illumination non-uniformity)
      - patient 14: left eye is a byte-identical duplicate of patient 1's left eye

    Returns (data_root, metadata_dataframe). The CSV is already written to
    data_root / "full_df.csv" so a cfg pointed at data_root can be fed
    straight into build_manifest, exercising the real adapter code path.
    """
    rng = np.random.default_rng(0)
    data_root = tmp_path
    image_dir = data_root / "preprocessed_images"
    image_dir.mkdir(parents=True)

    records: list[dict] = []

    def add_patient(pid, age, sex, normal, left_img, right_img, left_kw, right_kw):
        left_name = f"{pid}_left.jpg"
        right_name = f"{pid}_right.jpg"
        left_img.save(image_dir / left_name, quality=95)
        right_img.save(image_dir / right_name, quality=95)
        records.append(
            {
                "ID": pid,
                "Patient Age": age,
                "Patient Sex": sex,
                "Left-Fundus": left_name,
                "Right-Fundus": right_name,
                "Left-Diagnostic Keywords": left_kw,
                "Right-Diagnostic Keywords": right_kw,
                "N": 1 if normal else 0,
            }
        )

    normal_kw = "normal fundus"
    abnormal_kw = "moderate non proliferative retinopathy"

    for pid in range(1, 11):
        normal = pid <= 5
        img_l = _fundus_image(rng, brightness=190 if normal else 150)
        img_r = _fundus_image(rng, brightness=195 if normal else 145)
        kw = normal_kw if normal else abnormal_kw
        add_patient(
            pid,
            age=float(40 + pid) if pid != 5 else None,
            sex=("Male" if pid % 2 == 0 else "Female") if pid != 7 else None,
            normal=normal,
            left_img=img_l,
            right_img=img_r,
            left_kw=kw,
            right_kw=kw,
        )

    base_clean = _fundus_image(rng, brightness=190)

    blurred = base_clean.filter(ImageFilter.GaussianBlur(radius=6))
    add_patient(11, 55.0, "Male", True, base_clean, blurred, normal_kw, normal_kw)

    darkened = ImageEnhance.Brightness(base_clean).enhance(0.15)
    add_patient(12, 61.0, "Female", True, darkened, base_clean, normal_kw, normal_kw)

    dimmed = _fundus_image(rng, brightness=190, dim_left=True)
    add_patient(13, 47.0, "Male", False, dimmed, base_clean, abnormal_kw, normal_kw)

    duplicate_name = "14_left.jpg"
    shutil.copyfile(image_dir / "1_left.jpg", image_dir / duplicate_name)
    base_clean.save(image_dir / "14_right.jpg", quality=95)
    records.append(
        {
            "ID": 14,
            "Patient Age": 52.0,
            "Patient Sex": "Female",
            "Left-Fundus": duplicate_name,
            "Right-Fundus": "14_right.jpg",
            "Left-Diagnostic Keywords": normal_kw,
            "Right-Diagnostic Keywords": normal_kw,
            "N": 1,
        }
    )

    metadata = pd.DataFrame.from_records(records)
    metadata.to_csv(data_root / "full_df.csv", index=False)

    return data_root, metadata


@pytest.fixture
def synthetic_cfg(synthetic_fundus_dir, tmp_path):
    """Real default.yaml config, repointed at the synthetic fixture's temp dir."""
    data_root, _ = synthetic_fundus_dir
    cfg = load_config(
        overrides={
            "paths.data_root": str(data_root),
            "paths.artifacts": str(tmp_path / "artifacts"),
            "dataset.metadata_csv": "full_df.csv",
            "dataset.image_dir": "preprocessed_images",
        }
    )
    return cfg
