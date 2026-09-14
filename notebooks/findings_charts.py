"""Charts for the findings already established in README.md / docs/notes.md.

Descriptive analytics, not a new experiment. Every number here is either
recomputed directly from real artifacts (patient overlap, fellow-eye
concordance, arm results, cosine-similarity separation) or is a historical
measurement from a one-off investigation recorded in docs/notes.md, cited
inline where it is used. Several numbers fall in the second category
because the underlying training runs are not re-run in this session ("no
new experiments"): the first full-scale A/B run's per-seed AUROC
(superseded on disk by a later run that reused the same run-directory
names before train.py stopped overwriting them, still recorded in
docs/notes.md's "Full-scale A/B run" table), the train-set overlap
falsification test (a report-only investigation, never persisted as an
artifact), and arm E's post-patient-consistency-fix retrain plus the
leave-one-site-out sweep (both multi-hour, multi-seed training runs --
notebooks/arm_e_robustness_checks.py and notebooks/site4_followup.py,
full detail in docs/notes.md).

Same base64-inline pattern as eda.py and report.py; every section returns
(base64_png, takeaway) so report.py can reuse it directly.

Run: python notebooks/findings_charts.py
Output: artifacts/findings_report.html
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.dedupe import compute_resnet18_embeddings  # noqa: E402
from retinaprep.utils import load_current_run_metrics, load_run_index  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"

# Historical: first full-scale A/B run (cap=4473), docs/notes.md "Full-scale
# A/B run: 5 seeds, full dataset, GPU (cu126) -- real result", per-seed table.
# Superseded in artifacts/runs/ by the later 4-arm run, which reused the same
# run-directory names (A_seed42, ...) at a different cap (4435).
FIRST_RUN_SEEDS = [42, 43, 44, 45, 46]
FIRST_RUN_A_AUROC = [0.8151, 0.8155, 0.8105, 0.7984, 0.8097]
FIRST_RUN_B_AUROC = [0.7895, 0.7815, 0.7949, 0.7925, 0.8044]

# Historical: train-set overlap falsification test, docs/notes.md "Why
# curation costs AUROC" (b) and (c). B's and C's actual training sets
# (both size 4435) share 3071 images; removing 43 random,
# quality-uncorrelated images and re-splitting gives a near-identical
# overlap with B's original training set -- the control that shows the
# reshuffling isn't specific to which images quality curation removed.
TRAIN_OVERLAP_REAL = 3071 / 4435
TRAIN_OVERLAP_RANDOM_CONTROL = 0.694

# Historical: arm E (site_group split), retrained post patient-consistency-fix
# (docs/notes.md, "Follow-up C") -- notebooks/arm_e_robustness_checks.py,
# part1_prevalence_matched's "b_original"/"e_original", 5 seeds each. Not
# recomputed live for the same reason FIRST_RUN_*_AUROC above isn't: this is
# a multi-hour, multi-seed retraining sweep, not something to redo on every
# chart regen. Supersedes (by <0.002 AUROC, test fold unchanged) the
# pre-fix arm E cohort still on disk in artifacts/runs/index.json, whose
# config_hash doesn't change with the split file's content and can't
# distinguish the two on its own -- a known gap, noted rather than patched
# around by editing the index (docs/notes.md).
ARM_E_B_SEEDS = [42, 43, 44, 45, 46]
ARM_E_B_AUROC = [0.7835091403162056, 0.7887104743083003, 0.8028927865612647,
                 0.8009980237154151, 0.7815217391304347]
ARM_E_E_AUROC = [0.7419751711730594, 0.7299373821001885, 0.7443670090127855,
                 0.7334896597498777, 0.7352679819045622]
ARM_E_B_PREVALENCE = 0.5504300234558248
ARM_E_E_PREVALENCE = 0.6296670030272452

# Historical: leave-one-site-out sweep (docs/notes.md, "Follow-up C"/"D"),
# notebooks/arm_e_robustness_checks.py + notebooks/site4_followup.py. gap is
# mean AUROC (3 seeds) minus the 3-seed B baseline (mean 0.79170); se_gap
# combines Hanley-McNeil test-fold-size variance with seed-to-seed variance
# (site4_followup.py's total_variance) -- accounts for fold size, not just
# seed count, which is why these error bars are much wider than a plain
# 3-seed std would give.
LOSO_B_MEAN_AUROC_3SEED = 0.7917041337285902
LOSO_SITES = {
    "site_0": {"n_test": 1982, "gap": -0.05294427963324566, "se_gap": 0.0392},
    "site_1": {"n_test": 501, "gap": -0.033397373524508556, "se_gap": 0.0604},
    "site_2": {"n_test": 404, "gap": -0.03678769377522839, "se_gap": 0.0705},
    "site_3": {"n_test": 379, "gap": -0.033927470219016875, "se_gap": 0.0669},
    "site_4": {"n_test": 336, "gap": 0.02084353980306508, "se_gap": 0.0759},
}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fig_patient_overlap() -> tuple[str, str]:
    """42.0% of patients straddle image_random's folds; 0% straddle patient_group's."""
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")
    patient_of = manifest.set_index("image_path")["patient_id"]

    pcts = {}
    counts = {}
    for split_name in ("image_random", "patient_group"):
        with open(ARTIFACTS / "splits" / f"{split_name}.json") as fh:
            split = json.load(fh)
        fold_of_patient: dict[str, set[str]] = {}
        for fold, paths in split.items():
            for p in paths:
                pid = patient_of.get(p)
                fold_of_patient.setdefault(pid, set()).add(fold)
        n_straddle = sum(1 for folds in fold_of_patient.values() if len(folds) > 1)
        n_patients = len(fold_of_patient)
        counts[split_name] = (n_straddle, n_patients)
        pcts[split_name] = n_straddle / n_patients * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["image_random\n(the wrong way)", "patient_group\n(the right way)"]
    values = [pcts["image_random"], pcts["patient_group"]]
    ax.bar(labels, values, color=["#C44E52", "#55A868"])
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("% of patients on both sides of the split")
    ax.set_title("Patient-level split leakage")
    ax.set_ylim(0, max(values) * 1.25)
    fig.tight_layout()

    n_s, n_p = counts["image_random"]
    takeaway = (
        f"{n_s}/{n_p} patients ({pcts['image_random']:.1f}%) land on both sides of "
        f"image_random -- by construction, {pcts['patient_group']:.0f}% do under "
        f"patient_group. This is the simplest and most important chart in this "
        f"project: it is a count, not a statistic, and needs no significance test."
    )
    return _fig_to_base64(fig), takeaway


def fig_concordance() -> tuple[str, str]:
    """Fellow eyes share a label 77.8% of the time, vs a 50.5% chance floor."""
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")
    counts = manifest.groupby("patient_id").size()
    two_eye_ids = counts[counts == 2].index
    two_eye = manifest[manifest["patient_id"].isin(two_eye_ids)]
    agree = two_eye.groupby("patient_id")["label"].nunique()
    n_two_eye = len(two_eye_ids)
    n_concordant = int((agree == 1).sum())
    observed_pct = n_concordant / n_two_eye * 100

    balance = manifest["label"].value_counts(normalize=True)
    chance_pct = (balance.get(0, 0) ** 2 + balance.get(1, 0) ** 2) * 100

    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["Chance floor\n(class-balance only)", "Observed\n(fellow-eye pairs)"]
    values = [chance_pct, observed_pct]
    ax.bar(labels, values, color=["#8C8C8C", "#4C72B0"])
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("% concordant (same label)")
    ax.set_title(f"Fellow-eye label concordance (n={n_two_eye} two-eye patients)")
    ax.set_ylim(0, 100)
    fig.tight_layout()

    takeaway = (
        f"Fellow eyes share the same normal/abnormal label {observed_pct:.1f}% of the "
        f"time, against a {chance_pct:.1f}% floor from class balance alone -- real "
        f"bilateral correlation, but far short of 100%. This is why the ~42% patient "
        f"overlap shown above produces a modest ~0.017 AUROC effect rather than something "
        f"dramatic: leaking a patient's identity leaks a noisy, ~78%-reliable hint about "
        f"their label, not the label itself."
    )
    return _fig_to_base64(fig), takeaway


def _load_run_metrics() -> pd.DataFrame:
    """Per arm, only that arm's current cohort -- see
    retinaprep.utils.load_current_run_metrics. artifacts/runs/ no longer
    overwrites, so older cohorts (a previous config, an earlier point in
    the project's history) stay on disk and in artifacts/runs/index.json
    for provenance, but must not silently blend into these charts."""
    return load_current_run_metrics(ARTIFACTS)


def fig_arm_results() -> tuple[str, str]:
    """The 4-arm results, mean +/- std across seeds, with error bars."""
    df = _load_run_metrics()
    arms = ["A", "B", "C", "D"]
    arm_labels = {
        "A": "A\nimage_random",
        "B": "B\npatient_group",
        "C": "C\n+quality",
        "D": "D\n+quality+dedupe",
    }
    metrics = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("sens_95_spec", "Sens@95%Spec")]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharex=False)
    colors = ["#C44E52", "#55A868", "#4C72B0", "#8172B2"]
    for ax, (col, title) in zip(axes, metrics, strict=True):
        means = [df.loc[df["arm"] == a, col].mean() for a in arms]
        stds = [df.loc[df["arm"] == a, col].std(ddof=1) for a in arms]
        ax.bar([arm_labels[a] for a in arms], means, yerr=stds, capsize=4, color=colors)
        ax.set_title(title)
        ax.set_ylim(min(means) - max(stds) * 2 - 0.02, max(means) + max(stds) * 2 + 0.02)
    fig.suptitle("4-arm results (mean ± std across 5 seeds)")
    fig.tight_layout()

    b_auroc = df.loc[df["arm"] == "B", "auroc"]
    c_auroc = df.loc[df["arm"] == "C", "auroc"]
    d_auroc = df.loc[df["arm"] == "D", "auroc"]
    takeaway = (
        f"B (the correct split) scores {b_auroc.mean():.3f} AUROC vs A's "
        f"{df.loc[df['arm'] == 'A', 'auroc'].mean():.3f} -- leakage inflates the naive "
        f"split, as predicted. C is {c_auroc.mean() - b_auroc.mean():+.3f} vs B "
        f"(the surprising result: quality curation *cost* AUROC, sign-consistent across "
        f"all 5 seeds -- see docs/notes.md for why the flattering explanation didn't "
        f"survive being checked). D is {d_auroc.mean() - b_auroc.mean():+.3f} vs B, "
        f"consistent with the predicted null for a curation stage removing <1% of images."
    )
    return _fig_to_base64(fig), takeaway


def fig_ab_replication() -> tuple[str, str]:
    """The A-vs-B gap across three independent measurements -- same code,
    three different draws, one of them (Run 3) also under a different
    split-mode design (persisted_base, natural sizes) than the other two
    (recompute_per_arm, capped). Each bar pair is labeled by run, split
    mode, and training-set cap so no chart silently redefines what "Run 2"
    or "Run 3" means -- see docs/notes.md's provenance section for why
    Run 1's raw per-seed data has to come from FIRST_RUN_*_AUROC rather
    than from disk.
    """
    entries = pd.DataFrame(load_run_index(ARTIFACTS))
    # Two persisted cohorts distinguished by arm B's training-set size:
    # capped (4435, pre-fix, recompute_per_arm) vs natural (4474, post-fix,
    # persisted_base) -- distinguishing by n_train rather than hardcoding
    # a specific hash value, since the hash is just a fingerprint.
    b_rows = entries[entries["arm"] == "B"]
    capped_hash = b_rows.loc[b_rows["n_train"] < 4470, "config_hash"].iloc[0]
    natural_hash = b_rows.loc[b_rows["n_train"] >= 4470, "config_hash"].iloc[0]

    def _auroc(config_hash: str, arm: str) -> np.ndarray:
        sub = entries[(entries["config_hash"] == config_hash) & (entries["arm"] == arm)]
        return sub.sort_values("seed")["auroc"].to_numpy()

    run1_a, run1_b = np.array(FIRST_RUN_A_AUROC), np.array(FIRST_RUN_B_AUROC)
    run2_a, run2_b = _auroc(capped_hash, "A"), _auroc(capped_hash, "B")
    run3_a, run3_b = _auroc(natural_hash, "A"), _auroc(natural_hash, "B")

    runs = [
        ("Run 1\nrecompute_per_arm, cap=4473", run1_a, run1_b),
        ("Run 2\nrecompute_per_arm, cap=4435", run2_a, run2_b),
        ("Run 3\npersisted_base, uncapped", run3_a, run3_b),
    ]
    ps = [stats.ttest_rel(a, b)[1] for _, a, b in runs]
    diffs = [a.mean() - b.mean() for _, a, b in runs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    width = 0.32
    a_means = [a.mean() for _, a, _ in runs]
    a_stds = [a.std(ddof=1) for _, a, _ in runs]
    b_means = [b.mean() for _, _, b in runs]
    b_stds = [b.std(ddof=1) for _, _, b in runs]
    ax.bar(
        x - width / 2, a_means, width, yerr=a_stds, capsize=4,
        label="A (image_random)", color="#C44E52",
    )
    ax.bar(
        x + width / 2, b_means, width, yerr=b_stds, capsize=4,
        label="B (patient_group)", color="#55A868",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\np={p:.3f}" for (label, _, _), p in zip(runs, ps, strict=True)])
    ax.set_ylabel("AUROC")
    ax.set_title("A-vs-B gap: three independent 5-seed runs")
    ax.legend()
    fig.tight_layout()

    takeaway = (
        f"Run 1 (cap=4473): A-B={diffs[0]:+.4f}, p={ps[0]:.3f}. "
        f"Run 2 (cap=4435): A-B={diffs[1]:+.4f}, p={ps[1]:.3f}. "
        f"Run 3 (uncapped, post split-then-curate fix): A-B={diffs[2]:+.4f}, p={ps[2]:.3f}. "
        f"Runs 1 and 3 cluster near +0.018; Run 2 sits lower at +0.008 -- none reaches "
        f"significance except Run 1's uncorrected p, and none survives Bonferroni "
        f"correction across the family this project holds every claim to. Three draws, "
        f"one direction, no draw individually conclusive: this is what n=5 seeds being "
        f"underpowered looks like when actually re-drawn twice, not argued from theory once."
    )
    return _fig_to_base64(fig), takeaway


def fig_split_hierarchy() -> tuple[str, str]:
    """The three-rung hierarchy in one AUROC chart: image_random (arm A,
    leaks patients) vs patient_group (arm B, leaks sites) vs site_group
    (arm E, isolates the signal). Each bar labeled by arm, split mode, and
    seed count so a reader can't mistake this for one split evaluated three
    ways -- A and B share a test set (patient_group's), E does not (its
    test set is a single held-out site with different class balance,
    flagged explicitly rather than left implicit in the bar heights)."""
    df = _load_run_metrics()
    a_auroc = df.loc[df["arm"] == "A", "auroc"].to_numpy()
    b_auroc = df.loc[df["arm"] == "B", "auroc"].to_numpy()
    e_auroc = np.array(ARM_E_E_AUROC)

    labels = [
        "A\nimage_random\n(leaks patients)",
        "B\npatient_group\n(leaks sites)",
        "E\nsite_group\n(isolates signal)",
    ]
    means = [a_auroc.mean(), b_auroc.mean(), e_auroc.mean()]
    stds = [a_auroc.std(ddof=1), b_auroc.std(ddof=1), e_auroc.std(ddof=1)]
    colors = ["#C44E52", "#55A868", "#4C72B0"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors)
    for bar, m in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m + 0.015, f"{m:.3f}",
            ha="center", fontweight="bold",
        )
    ax.set_ylabel("AUROC (mean ± std, 5 seeds)")
    ax.set_title("Three-rung hierarchy: the deeper the split, the more leakage found")
    ax.set_ylim(0.65, 0.86)
    fig.tight_layout()

    ab_gap = means[0] - means[1]
    be_gap = means[2] - means[1]
    takeaway = (
        f"A (image_random, {a_auroc.mean():.3f}) vs B (patient_group, {b_auroc.mean():.3f}): "
        f"{ab_gap:+.3f} AUROC (A-B) -- real but small, and did not replicate significantly across "
        f"three independent 5-seed runs (chart above). B vs E (site_group, {e_auroc.mean():.3f}): "
        f"{be_gap:+.3f} AUROC (E-B) -- over 3x larger, 5/5 seeds agreeing, Bonferroni-significant "
        f"(p=0.0026). Not a like-for-like comparison of test sets: E's test fold is a single "
        f"held-out site with {ARM_E_E_PREVALENCE*100:.0f}% abnormal prevalence vs "
        f"B's {ARM_E_B_PREVALENCE*100:.0f}% -- checked directly (docs/notes.md) and the drop "
        f"survives prevalence-matching within 6% of its own size, so this is not a prevalence "
        f"artifact. Each rung's split catches leakage the rung above it structurally cannot."
    )
    return _fig_to_base64(fig), takeaway


def fig_leave_one_site_out() -> tuple[str, str]:
    """All five leave-one-site-out gaps, error bars included, site_4's
    positive exception shown exactly as measured -- not smoothed over.
    Error bars combine Hanley-McNeil test-fold-size variance with
    seed-to-seed variance (site4_followup.py), which is why every single
    interval spans zero even though 4/5 point estimates land in the
    predicted direction: this chart is deliberately showing both facts at
    once, not picking the flattering one."""
    sites = list(LOSO_SITES.keys())
    gaps = [LOSO_SITES[s]["gap"] for s in sites]
    ses = [LOSO_SITES[s]["se_gap"] for s in sites]
    n_tests = [LOSO_SITES[s]["n_test"] for s in sites]
    colors = ["#C44E52" if g < 0 else "#55A868" for g in gaps]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(sites))
    bars = ax.bar(x, gaps, yerr=[1.96 * se for se in ses], capsize=5, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n_test={n})" for s, n in zip(sites, n_tests, strict=True)])
    ax.set_ylabel("AUROC gap vs arm B (3-seed baseline)")
    ax.set_title("Leave-one-site-out: gap vs B, 95% CI (test-fold-size aware)")
    for bar, g in zip(bars, gaps, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, g + (0.012 if g >= 0 else -0.012),
            f"{g:+.3f}", ha="center", va="bottom" if g >= 0 else "top", fontweight="bold",
        )
    fig.tight_layout()

    n_negative = sum(1 for g in gaps if g < 0)
    takeaway = (
        f"{n_negative}/{len(sites)} held-out sites show the same negative gap as the original "
        f"arm E result (site_0) -- this is not just the one site that happened to land in the "
        f"original split. site_4 is a genuine exception ({LOSO_SITES['site_4']['gap']:+.3f}, "
        f"consistent across all 3 of its own seeds, not a training fluke) that class balance "
        f"and embedding-distance-from-training don't explain (docs/notes.md). Every interval "
        f"shown here spans zero, including site_0's -- accounting for test-fold size (not just "
        f"seed variance) shows no single site's estimate is precise enough alone to rule out a "
        f"true gap of zero; the sweep's value is the consistent direction across sites, not any "
        f"one site's number read in isolation."
    )
    return _fig_to_base64(fig), takeaway


def fig_train_overlap_falsification() -> tuple[str, str]:
    """Falsification test: does curation remove *specific* informative images, or
    does any small change to the input pool reshuffle StratifiedGroupKFold this much?"""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = [
        "B vs C\n(43 quality-rejected\nimages removed)",
        "B vs random control\n(43 random images removed)",
    ]
    values = [TRAIN_OVERLAP_REAL * 100, TRAIN_OVERLAP_RANDOM_CONTROL * 100]
    ax.bar(labels, values, color=["#4C72B0", "#8C8C8C"])
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("% overlap with B's original training set")
    ax.set_title("Training-set overlap: real removal vs random control")
    ax.set_ylim(0, 100)
    fig.tight_layout()

    takeaway = (
        f"Removing the 43 quality-rejected images and re-splitting gives "
        f"{TRAIN_OVERLAP_REAL*100:.1f}% overlap with B's original training set; "
        f"removing 43 *random*, quality-uncorrelated images gives "
        f"{TRAIN_OVERLAP_RANDOM_CONTROL*100:.1f}% -- statistically indistinguishable. "
        f"This is the falsification test for the flattering hypothesis (curation removed "
        f"informative-but-hard-to-grade images): it isn't about which images are removed, "
        f"it's that StratifiedGroupKFold reshuffles ~31% of the training set from almost "
        f"any small change to its input pool."
    )
    return _fig_to_base64(fig), takeaway


def fig_cosine_separation(cosine_min: float = 0.99) -> tuple[str, str]:
    """Cosine similarity: fellow-eye pairs vs confirmed duplicate pairs.

    Computed fresh from the real images (not resampled from docs/notes.md's
    summary stats), using the exact same embedding extraction dedupe.py
    uses (retinaprep.dedupe.compute_resnet18_embeddings, imported directly
    rather than re-derived here), so the separation shown here is directly
    checkable against the configured threshold rather than asserted.
    """
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")
    dup_df = pd.read_parquet(ARTIFACTS / "duplicates.parquet")

    paths = manifest["image_path"].tolist()
    embeddings = compute_resnet18_embeddings(paths)
    idx_of = {p: i for i, p in enumerate(paths)}

    counts = manifest.groupby("patient_id").size()
    two_eye_ids = counts[counts == 2].index
    fellow_sims = []
    for pid in two_eye_ids:
        rows = manifest.loc[manifest["patient_id"] == pid, "image_path"].tolist()
        i, j = idx_of[rows[0]], idx_of[rows[1]]
        fellow_sims.append(float(np.dot(embeddings[i], embeddings[j])))
    fellow_sims = np.array(fellow_sims)

    dup_sims = []
    dup_only = dup_df[dup_df["is_duplicate"]]
    for _cid, group in dup_only.groupby("cluster_id"):
        cluster_paths = group["image_path"].tolist()
        for a, b in itertools.combinations(cluster_paths, 2):
            dup_sims.append(float(np.dot(embeddings[idx_of[a]], embeddings[idx_of[b]])))
    dup_sims = np.array(dup_sims)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(fellow_sims.min(), dup_sims.min()) - 0.01, 1.0, 60)
    ax.hist(
        fellow_sims, bins=bins, alpha=0.6, density=True,
        label=f"fellow-eye pairs (n={len(fellow_sims)})", color="#4C72B0",
    )
    ax.hist(
        dup_sims, bins=bins, alpha=0.6, density=True,
        label=f"confirmed duplicates (n={len(dup_sims)})", color="#C44E52",
    )
    ax.axvline(cosine_min, color="black", linestyle="--", label=f"threshold ({cosine_min})")
    ax.set_xlabel("Cosine similarity (ResNet18 embeddings)")
    ax.set_ylabel("Density")
    ax.set_title("Embedding similarity: fellow eyes vs confirmed duplicates")
    ax.legend(fontsize=8)
    fig.tight_layout()

    below_threshold_dups = int((dup_sims < cosine_min).sum())
    above_threshold_fellow = int((fellow_sims >= cosine_min).sum())
    takeaway = (
        f"Fellow-eye pairs: mean {fellow_sims.mean():.3f}, max {fellow_sims.max():.3f} "
        f"(n={len(fellow_sims)}). Confirmed duplicates: mean {dup_sims.mean():.3f}, "
        f"min {dup_sims.min():.3f} (n={len(dup_sims)}). The threshold ({cosine_min}) "
        f"separates them almost completely -- {above_threshold_fellow} fellow-eye pairs "
        f"sit at or above it -- but not perfectly: {below_threshold_dups} confirmed "
        f"duplicate(s) fall below it, which is exactly why dedupe.py also needs phash "
        f"as an independent signal rather than relying on embeddings alone."
    )
    return _fig_to_base64(fig), takeaway


SECTION_TEMPLATE = """
<h3>{title}</h3>
<img src="data:image/png;base64,{img}">
<p class="takeaway"><strong>Takeaway:</strong> {takeaway}</p>
"""

PAGE_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>RetinaPrep Findings</title>
<style>
body {
  font-family: -apple-system, Segoe UI, sans-serif; max-width: 900px;
  margin: 2rem auto; padding: 0 1rem; color: #222;
}
h1, h3 { border-bottom: 2px solid #eee; padding-bottom: 0.3rem; }
img { max-width: 100%; }
.takeaway {
  background: #eef4fb; border-left: 4px solid #4C72B0; padding: 0.6rem 1rem;
  margin: 0.5rem 0 2rem 0;
}
</style></head>
<body>
<h1>RetinaPrep: leakage-audit findings</h1>
"""

PAGE_TAIL = "</body></html>"


def build_sections() -> list[tuple[str, str, str]]:
    sections = [
        ("Patient overlap across split strategies", fig_patient_overlap),
        ("Fellow-eye label concordance", fig_concordance),
        ("4-arm results", fig_arm_results),
        ("A-vs-B: does the gap replicate?", fig_ab_replication),
        ("The three-rung hierarchy: A vs B vs E", fig_split_hierarchy),
        ("Leave-one-site-out: is E's drop one site or a real pattern?", fig_leave_one_site_out),
        ("Training-set overlap falsification test", fig_train_overlap_falsification),
        ("Embedding similarity: fellow eyes vs duplicates", fig_cosine_separation),
    ]
    return [(title, *fn()) for title, fn in sections]


def main() -> None:
    sections = build_sections()
    html = [PAGE_HEAD]
    for title, img, takeaway in sections:
        html.append(SECTION_TEMPLATE.format(title=title, img=img, takeaway=takeaway))
    html.append(PAGE_TAIL)

    out_path = ARTIFACTS / "findings_report.html"
    out_path.write_text("".join(html), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
