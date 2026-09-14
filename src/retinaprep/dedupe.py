"""Step 6 — near-duplicate detection.

The sneakiest leakage source in hospital data: the same eye imaged twice in one
session, or a repeat visit, landing on both sides of a random split.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import pdist, squareform

from retinaprep.config import resolve_path
from retinaprep.utils import get_logger, save_json

logger = get_logger(__name__)

# phash alone has a high false-positive rate on fundus photography -- investigated
# directly (docs/notes.md): hamming<=6 (this project's configured default) flagged
# 13733 pairs on the real 6392-image dataset, the overwhelming majority visually
# confirmed false positives from shared macro-structure (dark background, circular
# FOV, similar framing), not real duplicate content. Every phash candidate is
# therefore verified by actual pixel difference before being trusted -- genuine
# duplicates measured ~0.00-0.01 on this data; the nearest false positive at any
# hamming distance measured 15+, so this cutoff has a wide, safe margin.
PIXEL_DIFF_VERIFIED_MAX = 5.0


def phash_duplicates(manifest: pd.DataFrame, hamming_max: int) -> list[tuple[str, str, int]]:
    """Perceptual-hash duplicate CANDIDATE pairs within hamming_max.

    Candidates, not verified duplicates -- see the module-level note on
    PIXEL_DIFF_VERIFIED_MAX. Always verify with `pixel_difference` before
    treating a phash match as real.
    """
    import imagehash

    paths = manifest["image_path"].tolist()
    hashes = []
    for p in paths:
        with Image.open(p) as im:
            hashes.append(imagehash.phash(im).hash.flatten())
    hash_matrix = np.array(hashes)

    frac = pdist(hash_matrix, metric="hamming")
    dist = squareform(frac * hash_matrix.shape[1])
    np.fill_diagonal(dist, 999)

    idx_i, idx_j = np.where(dist <= hamming_max)
    keep = idx_i < idx_j
    idx_i, idx_j = idx_i[keep], idx_j[keep]
    return [(paths[i], paths[j], int(dist[i, j])) for i, j in zip(idx_i, idx_j, strict=True)]


def pixel_difference(path_a: str, path_b: str) -> float:
    """Mean absolute per-pixel difference (0-255 scale), resizing to match if needed.

    The cheap, reliable check that separates a real duplicate from a phash
    coincidence -- see the module-level note.
    """
    with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
        rgb_a = im_a.convert("RGB")
        rgb_b = im_b.convert("RGB")
        if rgb_a.size != rgb_b.size:
            rgb_b = rgb_b.resize(rgb_a.size)
        arr_a = np.asarray(rgb_a, dtype=np.float64)
        arr_b = np.asarray(rgb_b, dtype=np.float64)
    return float(np.abs(arr_a - arr_b).mean())


def compute_resnet18_embeddings(paths: list[str], model_name: str = "resnet18") -> np.ndarray:
    """L2-normalized pretrained-ResNet18 penultimate features, one 512-dim row
    per path, in the given order.

    Factored out of `embedding_duplicates` so anything that needs the raw
    feature vectors (not just the near-duplicate pairs `embedding_duplicates`
    reports) uses the exact same extraction, not a re-derivation that could
    silently drift from it -- e.g. `notebooks/site_domain_distance.py`, which
    measures how far a held-out site's images sit from the training pool in
    this same feature space.
    """
    import torch
    from torchvision import models, transforms

    if model_name != "resnet18":
        raise ValueError(
            f"Unsupported dedupe.embedding_model {model_name!r}; only 'resnet18' is implemented"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    batch_size = 64
    features = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            tensors = []
            for p in batch_paths:
                with Image.open(p) as im:
                    tensors.append(transform(im.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            feats = torch.nn.functional.normalize(backbone(batch), dim=1)
            features.append(feats.cpu().numpy())
    return np.concatenate(features, axis=0)


def embedding_duplicates(
    manifest: pd.DataFrame, cosine_min: float, model_name: str
) -> list[tuple[str, str, float]]:
    """Near-duplicates via pretrained ResNet18 penultimate features + cosine similarity.

    phash catches crops and rescales; embeddings catch the same eye
    photographed twice under different illumination or exposure, where the
    raw pixels differ substantially even though the content is the same.
    """
    from sklearn.neighbors import NearestNeighbors

    paths = manifest["image_path"].tolist()
    feature_matrix = compute_resnet18_embeddings(paths, model_name)

    n_neighbors = min(6, len(paths))
    neighbours = NearestNeighbors(metric="cosine", n_neighbors=n_neighbors).fit(feature_matrix)
    distances, indices = neighbours.kneighbors(feature_matrix)

    pairs = []
    seen = set()
    for i in range(len(paths)):
        for dist, j in zip(distances[i], indices[i], strict=True):
            if i == j:
                continue
            similarity = 1.0 - dist
            if similarity >= cosine_min:
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    pairs.append((paths[i], paths[j], float(similarity)))
    return pairs


def deduplicate_across_methods(tagged_pairs: list[tuple]) -> list[tuple[str, str]]:
    """Collapse method-tagged candidate pairs (phash, embedding, ...) down to
    unique real-world pairs.

    A real duplicate pair either straddles a split boundary or it doesn't,
    regardless of how many detection methods caught it -- feeding
    method-tagged pairs straight into a count treats a pair found by both
    phash and embedding as two events instead of one, inflating any
    downstream aggregate (the money metric, or a "verified pairs total")
    by exactly the number of doubly-caught pairs. Always call this before
    counting or clustering across `verified_phash` + `embedding_pairs`
    combined -- see docs/notes.md for the bug this was written to catch.
    """
    seen = set()
    unique: list[tuple[str, str]] = []
    for a, b, *_ in tagged_pairs:
        key = (a, b) if a < b else (b, a)
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def cross_split_duplicate_count(pairs: list[tuple], split: dict) -> int:
    """The money metric: how many duplicate pairs straddle a given split's folds.

    This number is what makes arm A's inflated score concrete rather than
    theoretical -- reported prominently in run_dedupe.
    """
    fold_of = {p: fold for fold, paths in split.items() for p in paths}
    count = 0
    for a, b, *_ in pairs:
        fold_a, fold_b = fold_of.get(a), fold_of.get(b)
        if fold_a is not None and fold_b is not None and fold_a != fold_b:
            count += 1
    return count


def _assign_clusters(pairs: list[tuple]) -> dict[str, int]:
    """Union-find over verified duplicate pairs -> {image_path: cluster_id}."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b, *_ in pairs:
        union(a, b)

    roots = {p: find(p) for p in parent}
    root_to_cluster_id = {root: i for i, root in enumerate(sorted(set(roots.values())))}
    return {p: root_to_cluster_id[root] for p, root in roots.items()}


def run_dedupe(cfg: dict) -> None:
    """Find near-duplicates via phash (pixel-difference verified) + embeddings,
    report the cross-split money metric and any cross-patient integrity issue,
    and write artifacts/duplicates.parquet with cluster ids."""
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    manifest = pd.read_parquet(artifacts_dir / "manifest.parquet")
    dedupe_cfg = cfg["dedupe"]

    logger.info("Computing phash candidates (hamming<=%d)...", dedupe_cfg["phash_hamming_max"])
    phash_candidates = phash_duplicates(manifest, dedupe_cfg["phash_hamming_max"])
    logger.info("%d phash candidate pairs; verifying by pixel difference...", len(phash_candidates))

    verified_phash = [
        (a, b, hamming, pixel_difference(a, b)) for a, b, hamming in phash_candidates
    ]
    verified_phash = [t for t in verified_phash if t[3] < PIXEL_DIFF_VERIFIED_MAX]
    logger.info(
        "%d/%d phash candidates verified as genuine duplicates (pixel diff < %.1f)",
        len(verified_phash),
        len(phash_candidates),
        PIXEL_DIFF_VERIFIED_MAX,
    )

    logger.info(
        "Computing embedding candidates (cosine>=%.2f, model=%s)...",
        dedupe_cfg["embedding_cosine_min"],
        dedupe_cfg["embedding_model"],
    )
    embedding_pairs = embedding_duplicates(
        manifest, dedupe_cfg["embedding_cosine_min"], dedupe_cfg["embedding_model"]
    )
    logger.info("%d embedding candidate pairs", len(embedding_pairs))

    patient_by_path = manifest.set_index("image_path")["patient_id"]
    eye_by_path = manifest.set_index("image_path")["eye"]

    def _audit(pairs: list[tuple], name: str) -> None:
        for a, b, *_ in pairs:
            pa, pb = patient_by_path.get(a), patient_by_path.get(b)
            ea, eb = eye_by_path.get(a), eye_by_path.get(b)
            if pa == pb and ea != eb:
                logger.warning(
                    "%s flags fellow eyes as duplicate (should never happen): "
                    "patient %s, %s <-> %s -- threshold is too loose, raise it",
                    name,
                    pa,
                    a,
                    b,
                )
            elif pa != pb:
                logger.warning(
                    "%s: genuine dataset integrity issue -- same image under two "
                    "different patient IDs: patient %s (%s) vs patient %s (%s)",
                    name,
                    pa,
                    a,
                    pb,
                    b,
                )

    _audit(verified_phash, "phash")
    _audit(embedding_pairs, "embedding")

    all_verified = [(a, b, "phash", h) for a, b, h, _d in verified_phash] + [
        (a, b, "embedding", s) for a, b, s in embedding_pairs
    ]
    # A pair found by both methods is one real duplicate, not two -- every
    # cross-method aggregate below must run on the deduplicated union, not
    # the raw concatenation. See deduplicate_across_methods's docstring.
    unique_pairs = deduplicate_across_methods(all_verified)
    logger.info(
        "%d unique verified pairs (%d phash, %d embedding, %d found by both)",
        len(unique_pairs),
        len(verified_phash),
        len(embedding_pairs),
        len(verified_phash) + len(embedding_pairs) - len(unique_pairs),
    )

    splits_dir = artifacts_dir / "splits"
    cross_split_report = {}
    for split_name in ("image_random", "patient_group"):
        split_path = splits_dir / f"{split_name}.json"
        if not split_path.exists():
            continue
        with open(split_path) as fh:
            split = json.load(fh)
        n_straddle = cross_split_duplicate_count(unique_pairs, split)
        cross_split_report[split_name] = n_straddle
        logger.info(
            "Money metric: %d/%d verified duplicate pairs straddle the %s split",
            n_straddle,
            len(unique_pairs),
            split_name,
        )

    cluster_of = _assign_clusters(unique_pairs)
    dup_df = pd.DataFrame(
        {
            "image_path": manifest["image_path"],
            "patient_id": manifest["patient_id"],
            "cluster_id": manifest["image_path"].map(cluster_of),
        }
    )
    dup_df["is_duplicate"] = dup_df["cluster_id"].notna()

    dup_path = artifacts_dir / "duplicates.parquet"
    dup_df.to_parquet(dup_path, index=False)
    logger.info(
        "Wrote %s: %d images in %d duplicate clusters (%d verified pairs total)",
        dup_path,
        int(dup_df["is_duplicate"].sum()),
        dup_df["cluster_id"].nunique(),
        len(unique_pairs),
    )

    save_json(cross_split_report, artifacts_dir / "duplicates_cross_split.json")
