"""Six PNGs for README.md, exported to docs/figures/.

Designed for a README, not a notebook: readable at ~800px wide, large
fonts, minimal chartjunk, and a title that states the finding rather
than describing the axes. Reuses the same underlying data/logic as
notebooks/findings_charts.py (imported directly, not duplicated) but
with distinct, README-appropriate styling -- these are not the same
figures as the HTML reports.

Run: python notebooks/readme_figures.py
Output: docs/figures/{patient_overlap,concordance,ab_replication,
        split_hierarchy,leave_one_site_out,quality_examples}.png
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.utils import load_run_index  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
    }
)

BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
GREY = "#999999"


def _load_findings_module():
    spec = importlib.util.spec_from_file_location(
        "retinaprep_notebooks_findings_charts", REPO_ROOT / "notebooks" / "findings_charts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patient_overlap() -> None:
    import json

    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")
    patient_of = manifest.set_index("image_path")["patient_id"]

    pcts = {}
    for split_name in ("image_random", "patient_group"):
        with open(ARTIFACTS / "splits" / f"{split_name}.json") as fh:
            split = json.load(fh)
        fold_of_patient: dict[str, set[str]] = {}
        for fold, paths in split.items():
            for p in paths:
                pid = patient_of.get(p)
                fold_of_patient.setdefault(pid, set()).add(fold)
        n_straddle = sum(1 for folds in fold_of_patient.values() if len(folds) > 1)
        pcts[split_name] = n_straddle / len(fold_of_patient) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = ["Naive split\n(image_random)", "Correct split\n(patient_group)"]
    values = [pcts["image_random"], pcts["patient_group"]]
    bars = ax.bar(labels, values, color=[RED, GREEN], width=0.55)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 2.5, f"{v:.0f}%",
            ha="center", fontsize=22, fontweight="bold",
        )
    ax.set_ylabel("% of patients on both sides")
    ax.set_ylim(0, 55)
    ax.set_yticks([0, 25, 50])
    fig.suptitle(
        "42% of patients appear on both sides of a naive split", fontsize=19, y=1.02
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "patient_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def concordance() -> None:
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")
    counts = manifest.groupby("patient_id").size()
    two_eye_ids = counts[counts == 2].index
    two_eye = manifest[manifest["patient_id"].isin(two_eye_ids)]
    agree = two_eye.groupby("patient_id")["label"].nunique()
    observed_pct = (agree == 1).mean() * 100

    balance = manifest["label"].value_counts(normalize=True)
    chance_pct = (balance.get(0, 0) ** 2 + balance.get(1, 0) ** 2) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = ["By chance alone", "Observed\n(fellow eyes)"]
    values = [chance_pct, observed_pct]
    bars = ax.bar(labels, values, color=[GREY, BLUE], width=0.55)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 2.5, f"{v:.0f}%",
            ha="center", fontsize=22, fontweight="bold",
        )
    ax.set_ylabel("% sharing the same diagnosis")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    fig.suptitle(
        "Fellow eyes agree 78% of the time -- not 100%", fontsize=19, y=1.02
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "concordance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def ab_replication() -> None:
    """Three independent A-vs-B measurements, each labeled by run, split
    mode, and training-set cap -- Run 2 and Run 3 also differ in
    split-mode design (recompute_per_arm/capped vs persisted_base/
    natural), not just seed draw, so neither bar pair may be read as a
    plain repeat of the other without that label."""
    findings = _load_findings_module()
    run1_a, run1_b = np.array(findings.FIRST_RUN_A_AUROC), np.array(findings.FIRST_RUN_B_AUROC)

    entries = pd.DataFrame(load_run_index(ARTIFACTS))
    b_rows = entries[entries["arm"] == "B"]
    capped_hash = b_rows.loc[b_rows["n_train"] < 4470, "config_hash"].iloc[0]
    natural_hash = b_rows.loc[b_rows["n_train"] >= 4470, "config_hash"].iloc[0]

    def _auroc(config_hash: str, arm: str) -> np.ndarray:
        sub = entries[(entries["config_hash"] == config_hash) & (entries["arm"] == arm)]
        return sub.sort_values("seed")["auroc"].to_numpy()

    run2_a, run2_b = _auroc(capped_hash, "A"), _auroc(capped_hash, "B")
    run3_a, run3_b = _auroc(natural_hash, "A"), _auroc(natural_hash, "B")

    runs = [
        ("Run 1\ncap=4473", run1_a, run1_b),
        ("Run 2\ncap=4435", run2_a, run2_b),
        ("Run 3\nuncapped", run3_a, run3_b),
    ]
    ps = [stats.ttest_rel(a, b)[1] for _, a, b in runs]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(3)
    width = 0.3
    a_means = [a.mean() for _, a, _ in runs]
    a_stds = [a.std(ddof=1) for _, a, _ in runs]
    b_means = [b.mean() for _, _, b in runs]
    b_stds = [b.std(ddof=1) for _, _, b in runs]
    ax.bar(
        x - width / 2, a_means, width, yerr=a_stds, capsize=6, label="A (naive split)",
        color=RED, error_kw={"linewidth": 2},
    )
    ax.bar(
        x + width / 2, b_means, width, yerr=b_stds, capsize=6, label="B (correct split)",
        color=GREEN, error_kw={"linewidth": 2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\np={p:.2f}" for (label, _, _), p in zip(runs, ps, strict=True)]
    )
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.74, 0.86)
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle(
        "None of three A-vs-B runs reaches significance at n=5 seeds",
        fontsize=18, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ab_replication.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def split_hierarchy() -> None:
    """Three-rung hierarchy in one AUROC chart: A (image_random) vs B
    (patient_group) vs E (site_group). Reuses findings_charts.py's data
    constants directly (imported, not retyped) so this can't silently drift
    from the HTML report's numbers."""
    findings = _load_findings_module()
    df = findings._load_run_metrics()
    a_auroc = df.loc[df["arm"] == "A", "auroc"].to_numpy()
    b_auroc = df.loc[df["arm"] == "B", "auroc"].to_numpy()
    e_auroc = np.array(findings.ARM_E_E_AUROC)

    labels = ["A\nimage_random", "B\npatient_group", "E\nsite_group"]
    means = [a_auroc.mean(), b_auroc.mean(), e_auroc.mean()]
    stds = [a_auroc.std(ddof=1), b_auroc.std(ddof=1), e_auroc.std(ddof=1)]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(labels, means, yerr=stds, capsize=6, color=[RED, GREEN, BLUE], width=0.55)
    for bar, m in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m + 0.018, f"{m:.3f}",
            ha="center", fontsize=20, fontweight="bold",
        )
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.65, 0.87)
    fig.suptitle(
        "The deeper the split, the more leakage found: A > B > E", fontsize=19, y=1.02
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "split_hierarchy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def leave_one_site_out() -> None:
    """All five leave-one-site-out gaps, 95% CI included, site_4's positive
    exception shown exactly as measured. Reuses findings_charts.py's
    LOSO_SITES constant (imported, not retyped)."""
    findings = _load_findings_module()
    sites = list(findings.LOSO_SITES.keys())
    gaps = [findings.LOSO_SITES[s]["gap"] for s in sites]
    ses = [findings.LOSO_SITES[s]["se_gap"] for s in sites]
    n_tests = [findings.LOSO_SITES[s]["n_test"] for s in sites]
    colors = [RED if g < 0 else GREEN for g in gaps]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(sites))
    bars = ax.bar(x, gaps, yerr=[1.96 * se for se in ses], capsize=6, color=colors, width=0.6)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(sites, n_tests, strict=True)])
    ax.set_ylabel("AUROC gap vs arm B")
    for bar, g in zip(bars, gaps, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2, g + (0.018 if g >= 0 else -0.018), f"{g:+.3f}",
            ha="center", va="bottom" if g >= 0 else "top", fontsize=17, fontweight="bold",
        )
    fig.suptitle(
        "4/5 held-out sites replicate the drop -- site_4 doesn't", fontsize=19, y=1.02
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "leave_one_site_out.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def quality_examples() -> None:
    quality = pd.read_parquet(ARTIFACTS / "quality.parquet").sort_values("score")
    lowest = quality.head(6)
    highest = quality.tail(6)

    fig, axes = plt.subplots(2, 6, figsize=(13, 5.2))
    for row, (label, subset, color) in enumerate(
        [("6 lowest-scoring", lowest, RED), ("6 highest-scoring", highest, GREEN)]
    ):
        for col in range(6):
            ax = axes[row, col]
            path = subset.iloc[col]["image_path"]
            score = subset.iloc[col]["score"]
            with Image.open(path) as im:
                thumb = im.convert("RGB").resize((180, 180), Image.LANCZOS)
                ax.imshow(thumb)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2.5)
            ax.set_xlabel(f"{score:.2f}", fontsize=13)
        axes[row, 0].set_ylabel(label, fontsize=14, fontweight="bold")

    fig.suptitle(
        "Low scores are genuinely degraded; high scores are genuinely clean",
        fontsize=19, y=1.03,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(FIGURES_DIR / "quality_examples.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


FIGURE_NAMES = (
    "patient_overlap",
    "concordance",
    "ab_replication",
    "split_hierarchy",
    "leave_one_site_out",
    "quality_examples",
)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    patient_overlap()
    concordance()
    ab_replication()
    split_hierarchy()
    leave_one_site_out()
    quality_examples()
    for name in FIGURE_NAMES:
        path = FIGURES_DIR / f"{name}.png"
        print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
