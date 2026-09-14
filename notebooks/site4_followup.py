"""Two follow-ups on the leave-one-site-out sweep (notebooks/arm_e_robustness_checks.py):
site_4 was a genuine, unexplained positive-gap exception among 5 held-out sites.

A. Confidence interval on the E-vs-B gap per held-out site, accounting for
   test-set size -- not just seed-to-seed variance. site_4's test fold (336
   images) is the smallest of the five; if its interval comfortably spans
   zero while others don't, the sign flip may be within what a fold that
   size can resolve, rather than a firm opposite-direction result.

   Uses the closed-form Hanley-McNeil (1982) asymptotic SE for a single AUC
   estimate (needs only the AUC value and the fixed test fold's n1/n0 --
   no raw predictions required, so no retraining needed for this part).
   This asks a different, complementary question to the original 5-seed
   paired t-test: not "does this gap reproduce across differently-trained
   models on this SAME fixed test set" (already answered, decisively, in
   docs/notes.md) but "would this gap's sign and rough size survive
   swapping in a different, equally-sized sample from that site's
   population" -- test-set-size-driven generalization uncertainty, which
   the original seed-based analysis never quantified.

B. Mean pairwise embedding distance (pretrained ResNet18 features, same
   extraction as dedupe.compute_resnet18_embeddings) from each held-out
   site to the training pool (the rest of the dataset), ranked against the
   observed gap. Descriptive only at n=5 -- explicitly not a significance
   test. If distance-from-training predicts accuracy loss, that's the most
   practically useful thing in this project: a hospital could estimate
   expected accuracy loss at deployment by measuring how far their images
   sit from the training distribution, before ever deploying the model.

Run: python notebooks/site4_followup.py
Output: printed report + artifacts/site4_followup.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.dedupe import compute_resnet18_embeddings  # noqa: E402
from retinaprep.utils import get_logger  # noqa: E402

logger = get_logger(__name__)
ARTIFACTS = REPO_ROOT / "artifacts"

# From artifacts/arm_e_robustness_checks.json (part2_leave_one_site_out) --
# copied as literals rather than re-run, since re-running part 2 would mean
# retraining all 5 sites again for no new information.
SITE_AUROC = {
    "site_0": [0.7419751711730594, 0.7299373821001885, 0.7443670090127855],
    "site_1": [0.7666932397959183, 0.7668207908163265, 0.74140625],
    "site_2": [0.7546832491255344, 0.7471434123591137, 0.7629226583754372],
    "site_3": [0.740765502256393, 0.7731907070031756, 0.7593737812691514],
    "site_4": [0.826766304347826, 0.8126072654462243, 0.7982694508009153],
}
B_AUROC_3SEED = [0.7835091403162056, 0.7887104743083003, 0.8028927865612647]
SITES = list(SITE_AUROC.keys())


def hanley_mcneil_variance(auc: float, n1: int, n0: int) -> float:
    """Asymptotic sampling variance of an empirical AUC estimate (Hanley &
    McNeil, 1982), given only the AUC point estimate and the fixed test
    fold's positive/negative counts -- no raw predictions needed."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n0 - 1) * (q2 - (1 - auc) ** 2)) / (
        n1 * n0
    )


def total_variance(aurocs: list[float], n1: int, n0: int) -> tuple[float, float, float]:
    """Total sampling variance of the seed-mean AUC = test-set-size-driven
    Hanley-McNeil variance (same fixed test fold for every seed, so this
    term does NOT shrink with more seeds) + seed-to-seed training variance
    divided by n_seeds (this term does shrink -- more seeds, more averaged-
    out model-to-model noise). Returns (mean_auc, hm_var, seed_se_sq)."""
    mean_auc = float(np.mean(aurocs))
    hm_var = hanley_mcneil_variance(mean_auc, n1, n0)
    n = len(aurocs)
    seed_var = float(np.var(aurocs, ddof=1)) if n > 1 else 0.0
    seed_se_sq = seed_var / n
    return mean_auc, hm_var, seed_se_sq


def part_a_confidence_intervals(
    site_n1n0: dict[str, tuple[int, int]], b_n1: int, b_n0: int
) -> dict:
    b_mean, b_hm_var, b_seed_se_sq = total_variance(B_AUROC_3SEED, b_n1, b_n0)
    b_total_var = b_hm_var + b_seed_se_sq

    results = {
        "b_mean_auroc": b_mean,
        "b_hm_var": b_hm_var,
        "b_seed_se_sq": b_seed_se_sq,
        "b_total_var": b_total_var,
        "sites": {},
    }
    for site in SITES:
        n1, n0 = site_n1n0[site]
        mean_auc, hm_var, seed_se_sq = total_variance(SITE_AUROC[site], n1, n0)
        total_var = hm_var + seed_se_sq
        gap = mean_auc - b_mean
        se_gap = (total_var + b_total_var) ** 0.5
        ci_lo, ci_hi = gap - 1.96 * se_gap, gap + 1.96 * se_gap
        results["sites"][site] = {
            "n1": n1,
            "n0": n0,
            "mean_auroc": mean_auc,
            "hm_var": hm_var,
            "seed_se_sq": seed_se_sq,
            "total_var": total_var,
            "gap": gap,
            "se_gap": se_gap,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "ci_spans_zero": ci_lo <= 0 <= ci_hi,
        }
    return results


def part_b_embedding_distance(manifest: pd.DataFrame) -> dict:
    paths = manifest["image_path"].tolist()
    logger.info("Computing ResNet18 embeddings for %d images...", len(paths))
    embeddings = compute_resnet18_embeddings(paths)
    path_to_idx = {p: i for i, p in enumerate(paths)}

    distances = {}
    for site in SITES:
        site_paths = manifest.loc[manifest["site_label"] == site, "image_path"].tolist()
        pool_paths = manifest.loc[manifest["site_label"] != site, "image_path"].tolist()
        site_idx = [path_to_idx[p] for p in site_paths]
        pool_idx = [path_to_idx[p] for p in pool_paths]

        site_emb = embeddings[site_idx]
        pool_emb = embeddings[pool_idx]
        # Embeddings are L2-normalized (dedupe.compute_resnet18_embeddings),
        # so cosine similarity is a plain dot product; cosine distance = 1 - that.
        similarity = site_emb @ pool_emb.T
        mean_cosine_distance = float(1.0 - similarity.mean())
        distances[site] = {
            "n_site": len(site_idx),
            "n_pool": len(pool_idx),
            "mean_cosine_distance_to_pool": mean_cosine_distance,
        }
        logger.info(
            "%s: mean cosine distance to training pool = %.5f (n_site=%d, n_pool=%d)",
            site,
            mean_cosine_distance,
            len(site_idx),
            len(pool_idx),
        )
    return distances


def main() -> None:
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")
    site_labels = pd.read_parquet(ARTIFACTS / "site_labels.parquet")
    manifest = manifest.merge(site_labels, on="image_path")

    site_n1n0 = {}
    for site in SITES:
        sub = manifest[manifest["site_label"] == site]
        site_n1n0[site] = (int((sub["label"] == 1).sum()), int((sub["label"] == 0).sum()))
    # B's fixed test fold: n1=704, n0=575 (artifacts/runs/B_seed42_reeval_.../metrics.json,
    # test_metrics.n_positive=704, test_metrics.n=1279 -- patient_group split, unrelated
    # to site labels, so not derivable from the manifest+site_labels merge above).
    b_n1, b_n0 = 704, 575

    logger.info("=== Part A: confidence intervals on the E-vs-B gap, per site ===")
    part_a = part_a_confidence_intervals(site_n1n0, b_n1, b_n0)

    logger.info("=== Part B: embedding distance from each held-out site to the training pool ===")
    part_b = part_b_embedding_distance(manifest)

    gaps = {site: part_a["sites"][site]["gap"] for site in SITES}
    dists = {site: part_b[site]["mean_cosine_distance_to_pool"] for site in SITES}
    dist_array = np.array([dists[s] for s in SITES])
    gap_array = np.array([gaps[s] for s in SITES])
    pearson_r, pearson_p = stats.pearsonr(dist_array, gap_array)
    spearman_r, spearman_p = stats.spearmanr(dist_array, gap_array)

    result = {
        "part_a_confidence_intervals": part_a,
        "part_b_embedding_distance": part_b,
        "correlation": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "note": "n=5, descriptive only -- not a significance test",
        },
    }
    out_path = ARTIFACTS / "site4_followup.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print("\n--- Part A: CI on E-vs-B gap per site (Hanley-McNeil + seed variance) ---")
    for site in SITES:
        r = part_a["sites"][site]
        print(
            f"  {site}: gap={r['gap']:+.4f}  SE={r['se_gap']:.4f}  "
            f"95% CI=[{r['ci_95_lo']:+.4f}, {r['ci_95_hi']:+.4f}]  "
            f"spans_zero={r['ci_spans_zero']}"
        )

    print("\n--- Part B: embedding distance to training pool, ranked ---")
    for site in sorted(SITES, key=lambda s: dists[s]):
        print(f"  {site}: mean_cosine_distance={dists[site]:.5f}  gap={gaps[site]:+.4f}")
    print(
        f"\nPearson r={pearson_r:+.3f} (p={pearson_p:.3f}), "
        f"Spearman r={spearman_r:+.3f} (p={spearman_p:.3f}) -- n=5, descriptive only"
    )


if __name__ == "__main__":
    main()
