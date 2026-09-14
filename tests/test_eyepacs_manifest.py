"""The eyepacs adapter against the manifest contract -- the concrete test of
whether the adapter abstraction holds for a second dataset."""

import pandas as pd
import pytest

from retinaprep.adapters.eyepacs import build_manifest
from retinaprep.utils import MANIFEST_COLUMNS, validate_manifest


def test_manifest_has_required_columns(synthetic_eyepacs_dir, synthetic_eyepacs_cfg):
    _, metadata = synthetic_eyepacs_dir
    manifest = build_manifest(synthetic_eyepacs_cfg)

    assert list(manifest.columns) == MANIFEST_COLUMNS
    validate_manifest(manifest)  # must not raise

    assert len(manifest) == len(metadata)  # one manifest row per metadata row
    assert set(manifest["label"].unique()) <= {0, 1}
    assert (manifest["dataset_name"] == "eyepacs").all()
    assert manifest["age"].isna().all()  # EyePACS publishes no age/sex metadata
    assert manifest["sex"].isna().all()

    # Not every patient has two eyes -- patient 20 is single-eye by design.
    images_per_patient = manifest.groupby("patient_id").size()
    assert (images_per_patient == 1).sum() >= 1
    assert (images_per_patient == 2).sum() >= 1

    single_eye_rows = manifest[manifest["patient_id"] == "20"]
    assert len(single_eye_rows) == 1
    assert single_eye_rows.iloc[0]["eye"] == "L"


def test_label_from_referable_dr_threshold(synthetic_eyepacs_dir, synthetic_eyepacs_cfg):
    _, metadata = synthetic_eyepacs_dir
    manifest = build_manifest(synthetic_eyepacs_cfg)

    level_by_image = metadata.set_index("image")["level"]
    for _, row in manifest.iterrows():
        image_id = f"{row['patient_id']}_{'left' if row['eye'] == 'L' else 'right'}"
        expected = 1 if level_by_image[image_id] >= 2 else 0
        assert row["label"] == expected


def test_unrecognized_image_id_pattern_fails_loudly(synthetic_eyepacs_dir, synthetic_eyepacs_cfg):
    data_root, _ = synthetic_eyepacs_dir
    csv_path = data_root / "trainLabels.csv"
    raw = pd.read_csv(csv_path)
    raw.loc[raw.index[0], "image"] = "1_center"  # neither _left nor _right
    raw.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="do not match the expected"):
        build_manifest(synthetic_eyepacs_cfg)


def test_manifest_rejects_missing_image_paths(synthetic_eyepacs_dir, synthetic_eyepacs_cfg):
    manifest = build_manifest(synthetic_eyepacs_cfg)

    broken = manifest.copy()
    first = broken.index[0]
    broken.loc[first, "image_path"] = str(broken.loc[first, "image_path"]) + "_does_not_exist.jpeg"

    with pytest.raises(ValueError, match="nonexistent file"):
        validate_manifest(broken)


def test_eye_column_only_L_or_R(synthetic_eyepacs_dir, synthetic_eyepacs_cfg):
    manifest = build_manifest(synthetic_eyepacs_cfg)

    assert set(manifest["eye"].unique()) == {"L", "R"}

    broken = manifest.copy()
    broken.loc[broken.index[0], "eye"] = "left"

    with pytest.raises(ValueError, match="'eye' has values outside"):
        validate_manifest(broken)
