"""Step 6 — near-duplicate detection.

The sneakiest leakage source in hospital data: the same eye imaged twice in one
session, or a repeat visit, landing on both sides of a random split.
"""

from __future__ import annotations


def phash_duplicates(manifest, hamming_max: int):
    """Perceptual-hash duplicate pairs. TODO(claude-code): imagehash.phash."""
    raise NotImplementedError


def embedding_duplicates(manifest, cosine_min: float, model_name: str):
    """Near-duplicates via pretrained ResNet18 penultimate features + cosine similarity.

    TODO(claude-code): phash catches crops and rescales; embeddings catch the
    same eye photographed twice with different illumination, which phash misses.
    Use sklearn NearestNeighbors on L2-normalised features.
    """
    raise NotImplementedError


def cross_split_duplicate_count(pairs, split) -> int:
    """The money metric: how many duplicate pairs straddle a given split.

    TODO(claude-code): this number is what makes arm A's inflated score
    concrete rather than theoretical. Report it prominently.
    """
    raise NotImplementedError


def run_dedupe(cfg: dict) -> None:
    """TODO(claude-code): write artifacts/duplicates.parquet with cluster ids."""
    raise NotImplementedError
