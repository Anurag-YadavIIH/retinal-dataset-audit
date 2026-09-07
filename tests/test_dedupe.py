from retinaprep.adapters.odir5k import build_manifest
from retinaprep.dedupe import cross_split_duplicate_count, phash_duplicates, pixel_difference


def test_phash_finds_injected_exact_duplicate(synthetic_fundus_dir, synthetic_cfg):
    """Patient 14's left eye is a byte-identical copy of patient 1's left eye."""
    manifest = build_manifest(synthetic_cfg)

    candidates = phash_duplicates(manifest, hamming_max=6)

    path_1 = manifest.loc[manifest["image_path"].str.contains("1_left"), "image_path"].iloc[0]
    path_14 = manifest.loc[manifest["image_path"].str.contains("14_left"), "image_path"].iloc[0]

    found = any({a, b} == {path_1, path_14} for a, b, _hamming in candidates)
    assert found, "injected exact duplicate (1_left vs 14_left) not found by phash"

    assert pixel_difference(path_1, path_14) < 1.0


def test_cross_split_duplicate_count_is_reported():
    pairs = [
        ("a.jpg", "b.jpg", 0),  # straddles: train vs test
        ("c.jpg", "d.jpg", 0),  # same fold: both train
        ("e.jpg", "f.jpg", 0),  # one path not in the split at all -> not counted
    ]
    split = {
        "train": ["a.jpg", "c.jpg", "d.jpg"],
        "val": [],
        "test": ["b.jpg"],
    }

    assert cross_split_duplicate_count(pairs, split) == 1
