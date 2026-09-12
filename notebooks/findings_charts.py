"""Charts for the findings already established in README.md / docs/notes.md.

Descriptive analytics, not a new experiment. Every number here is either
recomputed directly from real artifacts (patient overlap, fellow-eye
concordance, arm results, cosine-similarity separation) or is a historical
measurement from a one-off investigation recorded in docs/notes.md, cited
inline where it is used. Two numbers fall in the second category because
the underlying training runs are not re-run in this session ("no new
experiments"), and their per-seed data predates artifacts/runs/index.json
(train.py used to overwrite same-named run directories; it no longer
does, see utils.append_run_index): the first full-scale A/B run's
per-seed AUROC (superseded on disk by a later run that reused the same
run-directory names before that fix landed, still recorded in
docs/notes.md's "Full-scale A/B run" table) and the train-set overlap
falsification test (a report-only investigation, never persisted as an
artifact).

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
from PIL import Image
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.utils import load_current_run_metrics  # noqa: E402

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
    """The A-vs-B gap in two separate 5-seed runs -- same code, different draw."""
    df = _load_run_metrics()
    second_a = df.loc[df["arm"] == "A", "auroc"].to_numpy()
    second_b = df.loc[df["arm"] == "B", "auroc"].to_numpy()
    first_a = np.array(FIRST_RUN_A_AUROC)
    first_b = np.array(FIRST_RUN_B_AUROC)

    _, p_first = stats.ttest_rel(first_a, first_b)
    _, p_second = stats.ttest_rel(second_a, second_b)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.32
    a_means = [first_a.mean(), second_a.mean()]
    a_stds = [first_a.std(ddof=1), second_a.std(ddof=1)]
    b_means = [first_b.mean(), second_b.mean()]
    b_stds = [first_b.std(ddof=1), second_b.std(ddof=1)]
    ax.bar(
        x - width / 2, a_means, width, yerr=a_stds, capsize=4,
        label="A (image_random)", color="#C44E52",
    )
    ax.bar(
        x + width / 2, b_means, width, yerr=b_stds, capsize=4,
        label="B (patient_group)", color="#55A868",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Run 1 (cap=4473)\np={p_first:.3f}", f"Run 2 (cap=4435)\np={p_second:.3f}"]
    )
    ax.set_ylabel("AUROC")
    ax.set_title("A-vs-B gap: two independent 5-seed runs")
    ax.legend()
    fig.tight_layout()

    diff1 = first_a.mean() - first_b.mean()
    diff2 = second_a.mean() - second_b.mean()
    takeaway = (
        f"Run 1: A-B = {diff1:+.4f} AUROC, paired t-test p={p_first:.3f}. Run 2, same "
        f"code and split strategies, different seed draw and training-set cap: "
        f"A-B = {diff2:+.4f}, p={p_second:.3f} -- the gap shrank by more than half and "
        f"lost significance. This is not a contradiction; it is what n=5 seeds being "
        f"underpowered looks like when actually re-drawn, not just argued from theory."
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


def _compute_embeddings(paths: list[str]) -> np.ndarray:
    """L2-normalized pretrained-ResNet18 penultimate features. Mirrors
    dedupe.embedding_duplicates's model/transform so this chart's cosine
    values are directly comparable to the threshold dedupe.py actually uses."""
    import torch
    from torchvision import models, transforms

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
            tensors = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(tensors).to(device)
            feats = torch.nn.functional.normalize(backbone(batch), dim=1)
            features.append(feats.cpu().numpy())
    return np.concatenate(features, axis=0)


def fig_cosine_separation(cosine_min: float = 0.99) -> tuple[str, str]:
    """Cosine similarity: fellow-eye pairs vs confirmed duplicate pairs.

    Computed fresh from the real images (not resampled from docs/notes.md's
    summary stats), using the same embedding extraction dedupe.py uses, so
    the separation shown here is directly checkable against the configured
    threshold rather than asserted.
    """
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")
    dup_df = pd.read_parquet(ARTIFACTS / "duplicates.parquet")

    paths = manifest["image_path"].tolist()
    embeddings = _compute_embeddings(paths)
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
