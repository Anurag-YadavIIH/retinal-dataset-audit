"""Quality metrics must move in the expected direction on known degradations."""

import cv2

from retinaprep.quality import (
    compute_metrics,
    detect_fov,
    exposure_metrics,
    gradability_score,
    illumination_uniformity,
    tenengrad,
    variance_of_laplacian,
)


def _load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_gray(path):
    return cv2.cvtColor(_load_rgb(path), cv2.COLOR_RGB2GRAY)


def test_blur_metric_drops_on_blurred_image(synthetic_fundus_dir):
    """Patient 11: left eye clean, right eye Gaussian-blurred."""
    data_root, _ = synthetic_fundus_dir
    image_dir = data_root / "preprocessed_images"

    clean = _load_gray(image_dir / "11_left.jpg")
    blurred = _load_gray(image_dir / "11_right.jpg")

    assert variance_of_laplacian(blurred) < variance_of_laplacian(clean)
    assert tenengrad(blurred) < tenengrad(clean)


def test_exposure_flags_darkened_image(synthetic_fundus_dir):
    """Patient 12: right eye clean, left eye darkened."""
    data_root, _ = synthetic_fundus_dir
    image_dir = data_root / "preprocessed_images"

    clean_rgb = _load_rgb(image_dir / "12_right.jpg")
    dark_rgb = _load_rgb(image_dir / "12_left.jpg")
    clean_mask, _, _ = detect_fov(clean_rgb)
    dark_mask, _, _ = detect_fov(dark_rgb)

    clean_exp = exposure_metrics(cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY), clean_mask)
    dark_exp = exposure_metrics(cv2.cvtColor(dark_rgb, cv2.COLOR_RGB2GRAY), dark_mask)

    assert dark_exp["mean_intensity"] < clean_exp["mean_intensity"]


def test_illumination_flags_dimmed_image(synthetic_fundus_dir):
    """Patient 13: left eye has its left half artificially dimmed (uneven flash)."""
    data_root, _ = synthetic_fundus_dir
    image_dir = data_root / "preprocessed_images"

    clean_rgb = _load_rgb(image_dir / "13_right.jpg")
    dimmed_rgb = _load_rgb(image_dir / "13_left.jpg")
    clean_mask, _, _ = detect_fov(clean_rgb)
    dimmed_mask, _, _ = detect_fov(dimmed_rgb)

    clean_cv = illumination_uniformity(cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2GRAY), clean_mask)
    dimmed_cv = illumination_uniformity(cv2.cvtColor(dimmed_rgb, cv2.COLOR_RGB2GRAY), dimmed_mask)

    assert dimmed_cv > clean_cv


def test_fov_detection_finds_circle(synthetic_fundus_dir):
    data_root, _ = synthetic_fundus_dir
    image_dir = data_root / "preprocessed_images"
    image = _load_rgb(image_dir / "1_left.jpg")

    mask, centre, radius = detect_fov(image)

    assert mask is not None
    assert radius > 0
    h, w = image.shape[:2]
    cx, cy = centre
    assert abs(cx - w / 2) < w * 0.2
    assert abs(cy - h / 2) < h * 0.2
    assert mask.shape == (h, w)
    assert mask.sum() > 0


def test_gradability_score_in_unit_range(synthetic_fundus_dir, synthetic_cfg):
    data_root, _ = synthetic_fundus_dir
    image_dir = data_root / "preprocessed_images"

    for name in ["1_left.jpg", "11_right.jpg", "12_left.jpg", "13_left.jpg"]:
        metrics = compute_metrics(_load_rgb(image_dir / name), synthetic_cfg)
        score = gradability_score(metrics, synthetic_cfg)
        assert 0.0 <= score <= 1.0

    clean_metrics = compute_metrics(_load_rgb(image_dir / "1_left.jpg"), synthetic_cfg)
    blurred_metrics = compute_metrics(_load_rgb(image_dir / "11_right.jpg"), synthetic_cfg)
    assert gradability_score(blurred_metrics, synthetic_cfg) < gradability_score(
        clean_metrics, synthetic_cfg
    )
