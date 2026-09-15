"""Segmentation adapters against the extended manifest contract.

Synthetic fixtures throughout -- CI must never need the real datasets,
same rule as the fundus adapters.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retinaprep.adapters.chasedb1 import build_manifest as chase_manifest
from retinaprep.adapters.chasedb1 import observer_mask_pairs
from retinaprep.adapters.idrid_seg import available_mask_classes
from retinaprep.adapters.idrid_seg import build_manifest as idrid_manifest
from retinaprep.config import load_config
from retinaprep.utils import MANIFEST_COLUMNS, validate_manifest


def _img(path, size=(32, 32), fill=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((*size, 3), fill, dtype=np.uint8), "RGB").save(path)


def _mask(path, size=(32, 32), on=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(size, dtype=np.uint8)
    if on:
        arr[10:20, 10:20] = 255
    Image.fromarray(arr, "L").save(path)


@pytest.fixture
def idrid_dir(tmp_path):
    """IDRiD layout: 3 train + 2 test images, optic disc for all, soft
    exudates for only some -- the absent-lesion case that must not be
    mistaken for a defect."""
    root = tmp_path / "idrid"
    for split, stems in {"a. Training Set": ["IDRiD_01", "IDRiD_02", "IDRiD_03"],
                         "b. Testing Set": ["IDRiD_55", "IDRiD_56"]}.items():
        for stem in stems:
            _img(root / "1. Original Images" / split / f"{stem}.jpg")
            _mask(root / "2. All Segmentation Groundtruths" / split / "5. Optic Disc"
                  / f"{stem}_OD.tif")
        # soft exudates only for the first stem of each split
        _mask(root / "2. All Segmentation Groundtruths" / split / "4. Soft Exudates"
              / f"{stems[0]}_SE.tif")
    return root


@pytest.fixture
def chase_dir(tmp_path):
    """CHASE_DB1 layout: 3 subjects x 2 eyes, both observers."""
    root = tmp_path / "chase"
    for subj in ("01", "02", "03"):
        for eye in ("L", "R"):
            stem = f"Image_{subj}{eye}"
            _img(root / "Images" / f"{stem}.jpg")
            for obs in ("1stHO", "2ndHO"):
                _mask(root / "Masks" / f"{stem}_{obs}.png")
    return root


def _cfg(root, name, **extra):
    # image_dir/mask_dir must be set per dataset: default.yaml's value is
    # ODIR-5K's ("preprocessed_images"), and every adapter in this project
    # overrides it, so inheriting it silently would point CHASE_DB1 at a
    # directory that does not exist.
    overrides = {"paths.data_root": str(root), "dataset.name": name}
    if name == "chasedb1":
        overrides.update({"dataset.image_dir": "Images", "dataset.mask_dir": "Masks"})
    overrides.update(extra)
    return load_config(overrides=overrides)


def test_idrid_optic_disc_manifest(idrid_dir):
    m = idrid_manifest(_cfg(idrid_dir, "idrid_seg"))
    assert list(m.columns) == [*MANIFEST_COLUMNS, "mask_path", "source_split"]
    validate_manifest(m, task="segmentation")  # must not raise
    assert len(m) == 5
    assert (m["source_split"] == "train").sum() == 3
    # eye and label are genuinely unknown for IDRiD, and that is allowed
    assert m["eye"].isna().all()
    assert m["label"].isna().all()


def test_idrid_absent_lesion_is_dropped_not_flagged(idrid_dir):
    """Soft exudates exist for 1 of 3 train and 1 of 2 test images. Those
    rows must be absent from the manifest, not present with a null or
    empty mask -- mis-handling this would make mask QC report false
    positives."""
    m = idrid_manifest(_cfg(idrid_dir, "idrid_seg", **{"dataset.mask_class": "SE"}))
    assert len(m) == 2
    assert m["mask_path"].notna().all()
    validate_manifest(m, task="segmentation")

    counts = available_mask_classes(idrid_dir)
    assert counts["OD"] == 5
    assert counts["SE"] == 2


def test_idrid_rejects_unknown_mask_class(idrid_dir):
    with pytest.raises(ValueError, match="Unknown mask_class"):
        idrid_manifest(_cfg(idrid_dir, "idrid_seg", **{"dataset.mask_class": "NOPE"}))


def test_chase_patient_grouping_is_non_vacuous(chase_dir):
    """The reason CHASE_DB1 is here beyond the second observer: two eyes
    per subject, so patient-level grouping differs from image-level."""
    m = chase_manifest(_cfg(chase_dir, "chasedb1"))
    validate_manifest(m, task="segmentation")
    assert len(m) == 6
    assert m["patient_id"].nunique() == 3
    assert set(m["eye"]) == {"L", "R"}
    assert (m.groupby("patient_id").size() == 2).all()


def test_chase_both_observers_pair_up(chase_dir):
    pairs = observer_mask_pairs(_cfg(chase_dir, "chasedb1"))
    assert len(pairs) == 6
    assert (pairs["mask_path_1stHO"] != pairs["mask_path_2ndHO"]).all()
    assert pairs["mask_path_1stHO"].str.endswith("_1stHO.png").all()
    assert pairs["mask_path_2ndHO"].str.endswith("_2ndHO.png").all()


def test_chase_fails_loudly_on_missing_observer(chase_dir):
    (chase_dir / "Masks" / "Image_01L_2ndHO.png").unlink()
    with pytest.raises(FileNotFoundError, match="inter-grader"):
        chase_manifest(_cfg(chase_dir, "chasedb1", **{"dataset.observer": "2ndHO"}))


def test_chase_fails_loudly_on_unexpected_filename(chase_dir):
    (chase_dir / "Images" / "Image_01L.jpg").rename(chase_dir / "Images" / "weird_name.jpg")
    with pytest.raises(ValueError, match="does not match the expected"):
        chase_manifest(_cfg(chase_dir, "chasedb1"))


def test_segmentation_validation_still_enforces_masks(chase_dir):
    m = chase_manifest(_cfg(chase_dir, "chasedb1"))
    broken = m.copy()
    broken.loc[broken.index[0], "mask_path"] = str(chase_dir / "Masks" / "does_not_exist.png")
    with pytest.raises(ValueError, match="mask_path.*nonexistent"):
        validate_manifest(broken, task="segmentation")

    missing = m.copy()
    missing.loc[missing.index[0], "mask_path"] = None
    with pytest.raises(ValueError, match="mask_path"):
        validate_manifest(missing, task="segmentation")


def test_classification_task_unaffected_by_the_extension(synthetic_cfg):
    """The fundus path must behave exactly as before: same required
    columns, same non-null rules, no mask_path needed."""
    from retinaprep.adapters.odir5k import build_manifest as odir_manifest

    m = odir_manifest(synthetic_cfg)
    assert list(m.columns) == MANIFEST_COLUMNS
    assert "mask_path" not in m.columns
    validate_manifest(m)                       # default task
    validate_manifest(m, task="classification")
    with pytest.raises(ValueError, match="mask_path"):
        validate_manifest(m, task="segmentation")


def test_unknown_task_rejected():
    with pytest.raises(ValueError, match="Unknown task"):
        validate_manifest(pd.DataFrame(), task="detection")
