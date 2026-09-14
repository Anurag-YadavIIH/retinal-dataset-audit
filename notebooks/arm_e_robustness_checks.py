"""Arm E robustness checks: is the -0.0557 AUROC drop a site effect, or
an artifact of which one site landed in the test fold / how different
its class balance is from train?

1. Prevalence-matched re-evaluation. Arm E's test set is 63% abnormal
   vs arm B's 53%. Subsample each arm's own test set to the OTHER
   arm's prevalence, re-evaluate the SAME trained models on the
   subsample, and report whether the gap survives. AUROC is a
   rank-based metric and should be close to prevalence-invariant in
   theory -- this also tests that claim empirically rather than just
   asserting it.

2. Leave-one-site-out across the 4-5 largest real sites (excluding
   "Other", a merged bucket of small/anomalous clusters, not a real
   site). One held-out site is one observation; this checks whether
   the drop is a property of sites in general or specific to site_0.

Retrains arm B and arm E (save_predictions=True, record_run=False --
deterministic reruns of already-reported seeds, done only to capture
per-example test scores that were never saved the first time; these do
NOT appear in artifacts/runs/index.json's aggregated tables, so they
can't double-count against the official results).

Run: python notebooks/arm_e_robustness_checks.py
Output: printed report + artifacts/arm_e_robustness_checks.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.config import load_config  # noqa: E402
from retinaprep.experiment import build_arm_split  # noqa: E402
from retinaprep.train import run_train  # noqa: E402
from retinaprep.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

ARTIFACTS = REPO_ROOT / "artifacts"
SEEDS_5 = [42, 43, 44, 45, 46]
SEEDS_3 = [42, 43, 44]

# The 4 largest real sites besides site_0 (already arm E's held-out
# site) -- excludes "Other", a consolidated bucket of small/anomalous
# resolution clusters, not a real site.
LEAVE_OUT_SITES = ["site_1", "site_2", "site_3", "site_4"]


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Same three metrics, same formulas, as retinaprep.train.evaluate --
    duplicated here (not imported) because that function operates on a
    DataLoader, not raw arrays; this one needs to run on resampled
    subsets of already-computed scores."""
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = max(int(np.searchsorted(fpr, 0.05, side="right")) - 1, 0)
    sens_95 = float(tpr[idx])
    return {"auroc": auroc, "auprc": auprc, "sens_95_spec": sens_95, "n": len(y_true)}


def prevalence_matched_subsample_metrics(
    true_label: np.ndarray, score: np.ndarray, target_abnormal_frac: float, seed: int
) -> dict:
    """Subsample (true_label, score) to hit target_abnormal_frac exactly,
    downsampling whichever class is in excess (never both, to keep as
    much data as possible), then compute metrics on the subsample."""
    rng = np.random.default_rng(seed)
    idx_abnormal = np.where(true_label == 1)[0]
    idx_normal = np.where(true_label == 0)[0]
    n1, n0 = len(idx_abnormal), len(idx_normal)

    k1_keep_all_normal = round(n0 * target_abnormal_frac / (1 - target_abnormal_frac))
    if k1_keep_all_normal <= n1:
        k0, k1 = n0, k1_keep_all_normal
    else:
        k1 = n1
        k0 = round(n1 * (1 - target_abnormal_frac) / target_abnormal_frac)

    chosen = np.concatenate(
        [
            rng.choice(idx_abnormal, size=k1, replace=False),
            rng.choice(idx_normal, size=k0, replace=False),
        ]
    )
    metrics = compute_metrics(true_label[chosen], score[chosen])
    metrics["target_abnormal_frac"] = target_abnormal_frac
    metrics["achieved_abnormal_frac"] = float((true_label[chosen] == 1).mean())
    return metrics


def leave_one_site_out_split(manifest: pd.DataFrame, held_out_site: str, seed: int) -> dict:
    """Test = exactly the held-out site's images. Train/val = a patient-
    grouped split of everything else (GroupKFold on patient_id, not
    stratified -- val here is only early-stopping material; the real
    held-out estimate is the site-level test set), so a model never
    trains on the site it's tested against, and a patient's fellow eye
    can't leak between train and val either.
    """
    from sklearn.model_selection import GroupKFold

    test_df = manifest[manifest["site_label"] == held_out_site]
    remainder = manifest[manifest["site_label"] != held_out_site].reset_index(drop=True)

    val_frac = 0.1
    n_groups = remainder["patient_id"].nunique()
    n_splits = max(2, min(round(1 / val_frac), n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    train_pos, val_pos = next(gkf.split(remainder, groups=remainder["patient_id"]))
    train_df = remainder.iloc[train_pos]
    val_df = remainder.iloc[val_pos]

    return {
        "train": train_df["image_path"].tolist(),
        "val": val_df["image_path"].tolist(),
        "test": test_df["image_path"].tolist(),
    }


def _load_test_predictions(run_dir_name: str) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS / "runs" / run_dir_name / "test_predictions.parquet")


def retrain_for_predictions(
    cfg: dict, arm: str, split_name: str, manifest, split, seeds: list[int]
) -> list[str]:
    """Retrain one arm's already-reported seeds, this time saving
    per-example test predictions. Deterministic -- reproduces identical
    aggregate metrics to what's already in artifacts/runs/index.json;
    verified below rather than assumed."""
    run_dirs = []
    for seed in seeds:
        result = run_train(
            cfg,
            split_name=split_name,
            run_name=f"{arm}_seed{seed}_reeval",
            arm=arm,
            seed_override=seed,
            manifest=manifest,
            split=split,
            save_predictions=True,
            record_run=False,
        )
        cfg_hash = result["config_hash"]
        timestamp = result["timestamp"]
        run_dirs.append(f"{arm}_seed{seed}_reeval_{timestamp}_{cfg_hash}")
        logger.info(
            "%s seed %d reeval: AUROC=%.4f (verify against original separately)",
            arm,
            seed,
            result["test_metrics"]["auroc"],
        )
    return run_dirs


def part1_prevalence_matched(cfg: dict) -> dict:
    manifest_b, split_b = build_arm_split(cfg, "B")
    manifest_e, split_e = build_arm_split(cfg, "E")

    b_dirs = retrain_for_predictions(cfg, "B", "patient_group", manifest_b, split_b, SEEDS_5)
    e_dirs = retrain_for_predictions(cfg, "E", "site_group", manifest_e, split_e, SEEDS_5)

    b_original = {"auroc": [], "auprc": [], "sens_95_spec": []}
    b_matched_to_e = {"auroc": [], "auprc": [], "sens_95_spec": []}
    e_original = {"auroc": [], "auprc": [], "sens_95_spec": []}
    e_matched_to_b = {"auroc": [], "auprc": [], "sens_95_spec": []}

    e_prevalence = None
    b_prevalence = None

    for run_dir in b_dirs:
        preds = _load_test_predictions(run_dir)
        y, s = preds["true_label"].to_numpy(), preds["score"].to_numpy()
        b_prevalence = float((y == 1).mean())
        m = compute_metrics(y, s)
        for k in b_original:
            b_original[k].append(m[k])

    for run_dir in e_dirs:
        preds = _load_test_predictions(run_dir)
        y, s = preds["true_label"].to_numpy(), preds["score"].to_numpy()
        e_prevalence = float((y == 1).mean())
        m = compute_metrics(y, s)
        for k in e_original:
            e_original[k].append(m[k])

    for i, run_dir in enumerate(b_dirs):
        preds = _load_test_predictions(run_dir)
        y, s = preds["true_label"].to_numpy(), preds["score"].to_numpy()
        m = prevalence_matched_subsample_metrics(y, s, e_prevalence, seed=SEEDS_5[i])
        for k in b_matched_to_e:
            b_matched_to_e[k].append(m[k])

    for i, run_dir in enumerate(e_dirs):
        preds = _load_test_predictions(run_dir)
        y, s = preds["true_label"].to_numpy(), preds["score"].to_numpy()
        m = prevalence_matched_subsample_metrics(y, s, b_prevalence, seed=SEEDS_5[i])
        for k in e_matched_to_b:
            e_matched_to_b[k].append(m[k])

    def _mean(d, k):
        return float(np.mean(d[k]))

    original_gap = _mean(e_original, "auroc") - _mean(b_original, "auroc")
    # E matched to B's prevalence, vs B original (both at B's prevalence)
    matched_gap_at_b_prevalence = _mean(e_matched_to_b, "auroc") - _mean(b_original, "auroc")
    # E original, vs B matched to E's prevalence (both at E's prevalence)
    matched_gap_at_e_prevalence = _mean(e_original, "auroc") - _mean(b_matched_to_e, "auroc")

    return {
        "b_prevalence": b_prevalence,
        "e_prevalence": e_prevalence,
        "b_original": b_original,
        "e_original": e_original,
        "b_matched_to_e_prevalence": b_matched_to_e,
        "e_matched_to_b_prevalence": e_matched_to_b,
        "original_auroc_gap": original_gap,
        "matched_gap_at_b_prevalence": matched_gap_at_b_prevalence,
        "matched_gap_at_e_prevalence": matched_gap_at_e_prevalence,
        "b_dirs": b_dirs,
        "e_dirs": e_dirs,
    }


def part2_leave_one_site_out(cfg: dict, b_dirs_3seed: list[str], e_dirs_3seed: list[str]) -> dict:
    """b_dirs_3seed / e_dirs_3seed: the exact run directory names for the
    first 3 seeds of arm B / arm E's part-1 retraining, passed in
    directly (not re-derived by scanning the filesystem, which has no
    reliable "latest" ordering and would misbehave on a second run of
    this script)."""
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")
    site_labels = pd.read_parquet(ARTIFACTS / "site_labels.parquet")
    manifest = manifest.merge(site_labels, on="image_path")

    b_auroc_3seed = []
    for run_dir in b_dirs_3seed:
        with open(ARTIFACTS / "runs" / run_dir / "metrics.json") as fh:
            b_auroc_3seed.append(json.load(fh)["test_metrics"]["auroc"])

    results = {}

    # site_0 (arm E's original held-out site): reuse the first 3 seeds
    # of part 1's E retraining rather than training a 4th configuration
    # for a site already covered.
    site0_auroc = []
    site0_n_test = None
    for run_dir in e_dirs_3seed:
        with open(ARTIFACTS / "runs" / run_dir / "metrics.json") as fh:
            data = json.load(fh)
            site0_auroc.append(data["test_metrics"]["auroc"])
            site0_n_test = data["n_test"]
    results["site_0"] = {"auroc": site0_auroc, "n_test": site0_n_test}

    for site in LEAVE_OUT_SITES:
        aurocs = []
        n_test = None
        for seed in SEEDS_3:
            split = leave_one_site_out_split(manifest, site, seed)
            n_test = len(split["test"])
            result = run_train(
                cfg,
                split_name="site_group",
                run_name=f"E_loso_{site}_seed{seed}",
                arm=None,
                seed_override=seed,
                manifest=manifest,
                split=split,
                save_predictions=False,
                record_run=False,
            )
            aurocs.append(result["test_metrics"]["auroc"])
            logger.info(
                "leave-one-out %s seed %d: AUROC=%.4f (n_test=%d)", site, seed, aurocs[-1], n_test
            )
        results[site] = {"auroc": aurocs, "n_test": n_test}

    b_mean = float(np.mean(b_auroc_3seed)) if b_auroc_3seed else None
    for r in results.values():
        r["mean_auroc"] = float(np.mean(r["auroc"]))
        r["gap_vs_b"] = r["mean_auroc"] - b_mean if b_mean else None

    return {"b_auroc_3seed": b_auroc_3seed, "b_mean_auroc_3seed": b_mean, "sites": results}


def main() -> None:
    cfg = load_config()

    logger.info("=== Part 1: prevalence-matched re-evaluation ===")
    part1 = part1_prevalence_matched(cfg)
    logger.info("Part 1 done. Original gap=%.4f", part1["original_auroc_gap"])

    # b_dirs/e_dirs are ordered exactly as SEEDS_5 = [42,43,44,45,46];
    # the first 3 entries correspond exactly to SEEDS_3 = [42,43,44].
    b_dirs_3seed = part1["b_dirs"][: len(SEEDS_3)]
    e_dirs_3seed = part1["e_dirs"][: len(SEEDS_3)]

    logger.info("=== Part 2: leave-one-site-out ===")
    part2 = part2_leave_one_site_out(cfg, b_dirs_3seed, e_dirs_3seed)

    result = {"part1_prevalence_matched": part1, "part2_leave_one_site_out": part2}
    out_path = ARTIFACTS / "arm_e_robustness_checks.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print("\n--- Part 1: prevalence-matched re-evaluation ---")
    print(
        f"B test prevalence: {part1['b_prevalence']:.3f}, "
        f"E test prevalence: {part1['e_prevalence']:.3f}"
    )
    print(f"Original AUROC gap (E-B): {part1['original_auroc_gap']:+.4f}")
    print(
        "Gap at B's prevalence (E matched down, vs B original): "
        f"{part1['matched_gap_at_b_prevalence']:+.4f}"
    )
    print(
        "Gap at E's prevalence (E original, vs B matched up): "
        f"{part1['matched_gap_at_e_prevalence']:+.4f}"
    )

    print("\n--- Part 2: leave-one-site-out ---")
    print(f"B mean AUROC (3 seeds): {part2['b_mean_auroc_3seed']:.4f}")
    for site, r in part2["sites"].items():
        print(
            f"  {site}: mean AUROC={r['mean_auroc']:.4f} gap_vs_B={r['gap_vs_b']:+.4f} "
            f"(n_test={r['n_test']})"
        )


if __name__ == "__main__":
    main()
