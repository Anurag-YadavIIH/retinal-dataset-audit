from retinaprep.adapters.odir5k import build_manifest
from retinaprep.dedupe import (
    cross_split_duplicate_count,
    deduplicate_across_methods,
    phash_duplicates,
    pixel_difference,
)


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


def test_deduplicate_across_methods_collapses_pair_found_by_both():
    """A pair phash AND embedding both flag must count once, not twice."""
    tagged_pairs = [
        ("a.jpg", "b.jpg", "phash", 0),
        ("a.jpg", "b.jpg", "embedding", 0.99),  # same real pair, other method
        ("c.jpg", "d.jpg", "phash", 2),
        ("e.jpg", "f.jpg", "embedding", 0.995),
    ]

    unique = deduplicate_across_methods(tagged_pairs)

    assert len(unique) == 3
    assert {"a.jpg", "b.jpg"} in [set(p) for p in unique]


def test_deduplicate_across_methods_ignores_pair_order():
    """(a, b) and (b, a) are the same real pair regardless of method order."""
    tagged_pairs = [
        ("a.jpg", "b.jpg", "phash", 0),
        ("b.jpg", "a.jpg", "embedding", 0.99),
    ]

    assert len(deduplicate_across_methods(tagged_pairs)) == 1


def test_straddle_count_does_not_double_count_a_pair_found_by_both_methods():
    """This is exactly the bug found in run_dedupe: a pair straddling a split
    boundary must count once in the money metric, no matter how many
    detection methods independently caught it. Feeding method-tagged pairs
    straight into cross_split_duplicate_count (skipping the dedupe step)
    reproduces the bug this test guards against.
    """
    tagged_pairs = [
        ("a.jpg", "b.jpg", "phash", 0),
        ("a.jpg", "b.jpg", "embedding", 0.99),  # same pair, found by both
        ("c.jpg", "d.jpg", "phash", 1),  # same fold, doesn't straddle
    ]
    split = {
        "train": ["a.jpg", "c.jpg", "d.jpg"],
        "val": [],
        "test": ["b.jpg"],
    }

    buggy_count = cross_split_duplicate_count(
        [(a, b) for a, b, *_ in tagged_pairs], split
    )
    assert buggy_count == 2, "sanity check: confirms the bug shape exists if not deduplicated first"

    unique_pairs = deduplicate_across_methods(tagged_pairs)
    fixed_count = cross_split_duplicate_count(unique_pairs, split)
    assert fixed_count == 1
