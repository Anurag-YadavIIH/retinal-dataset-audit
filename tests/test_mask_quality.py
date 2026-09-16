"""Mask QC detectors, verified by injecting the defects they exist to catch.

IDRiD's masks come back 0% rejected. That number only means "the data is
clean" if the checks are known to fire on dirty data -- otherwise it is
indistinguishable from "the checks are inert". Each test below builds a
specific defect and asserts the matching detector trips.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retinaprep.mask_quality import compute_mask_metrics, flag_masks, load_binary_mask


def _write_image(path, size=(64, 64), dark_border=True):
    """Fundus-like: bright disc on a black surround, as real images are."""
    arr = np.zeros((*size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    r = min(size) // 2 - 2
    inside = (xx - size[1] // 2) ** 2 + (yy - size[0] // 2) ** 2 <= r**2
    arr[inside] = 140
    if not dark_border:
        arr[:] = 140
    Image.fromarray(arr, "RGB").save(path)
    return path


def _write_mask(path, size=(64, 64), boxes=((28, 36, 28, 36),), value=255, mode="L"):
    arr = np.zeros(size, dtype=np.uint8)
    for y0, y1, x0, x1 in boxes:
        arr[y0:y1, x0:x1] = value
    Image.fromarray(arr, mode="L").convert(mode).save(path)
    return path


def _metrics(tmp_path, *, mask_kw=None, image_kw=None, single=True):
    img = _write_image(tmp_path / "im.png", **(image_kw or {}))
    msk = _write_mask(tmp_path / "mk.png", **(mask_kw or {}))
    return compute_mask_metrics(str(img), str(msk), expect_single_component=single)


def test_clean_mask_passes_everything(tmp_path):
    m = _metrics(tmp_path)
    row = flag_masks(pd.DataFrame([m])).iloc[0]
    assert row["dims_match"]
    assert not row["rejected"]
    assert row["n_components"] == 1
    assert row["reasons"] == ""


def test_dimension_mismatch_is_rejected(tmp_path):
    img = _write_image(tmp_path / "im.png", size=(64, 64))
    msk = _write_mask(tmp_path / "mk.png", size=(48, 48))
    m = compute_mask_metrics(str(img), str(msk), expect_single_component=True)
    assert not m["dims_match"]
    assert flag_masks(pd.DataFrame([m])).iloc[0]["reject_dim_mismatch"]


def test_empty_mask_is_rejected(tmp_path):
    m = _metrics(tmp_path, mask_kw={"boxes": ()})
    out = flag_masks(pd.DataFrame([m])).iloc[0]
    assert m["fg_pixels"] == 0
    assert out["reject_empty"] and out["rejected"]


def test_multi_component_flagged_only_when_single_expected(tmp_path):
    two = {"boxes": ((10, 16, 10, 16), (40, 46, 40, 46))}
    disc = _metrics(tmp_path, mask_kw=two, single=True)
    assert disc["n_components"] == 2
    assert flag_masks(pd.DataFrame([disc])).iloc[0]["flag_multi_component"]

    # the same geometry is legitimate for a lesion class
    lesion = _metrics(tmp_path, mask_kw=two, single=False)
    assert not flag_masks(pd.DataFrame([lesion])).iloc[0]["flag_multi_component"]


def test_misalignment_detected_via_foreground_on_black_surround(tmp_path):
    # annotation parked in the corner, i.e. off the retina entirely
    m = _metrics(tmp_path, mask_kw={"boxes": ((0, 8, 0, 8),)})
    assert m["fg_on_dark_fraction"] > 0.9
    assert flag_masks(pd.DataFrame([m])).iloc[0]["flag_misaligned"]


def test_area_outlier_flagged_against_the_run_distribution(tmp_path):
    rows = []
    for i in range(30):  # a tight, realistic area distribution
        img = _write_image(tmp_path / f"i{i}.png")
        msk = _write_mask(tmp_path / f"m{i}.png", boxes=((28, 36 + (i % 2), 28, 36),))
        rows.append(compute_mask_metrics(str(img), str(msk), True))
    img = _write_image(tmp_path / "big_i.png")
    msk = _write_mask(tmp_path / "big_m.png", boxes=((8, 56, 8, 56),))  # far larger
    rows.append(compute_mask_metrics(str(img), str(msk), True))

    out = flag_masks(pd.DataFrame(rows))
    assert out.iloc[-1]["flag_area_outlier"], "the oversized mask should be the outlier"
    assert not out.iloc[:-1]["flag_area_outlier"].any()


@pytest.mark.parametrize("mode,value", [("L", 255), ("P", 1)])
def test_load_binary_mask_handles_both_encodings(tmp_path, mode, value):
    """DRIVE ships observer 1 as mode 'L' {0,255} and observer 2 as mode
    'P' {0,1}. Assuming 255 means foreground reads the second observer as
    ~0% vessel, so this must not depend on encoding."""
    p = _write_mask(tmp_path / f"m_{mode}.png", value=value, mode=mode)
    fg = load_binary_mask(str(p))
    assert fg.dtype == bool
    assert fg.sum() == 8 * 8
