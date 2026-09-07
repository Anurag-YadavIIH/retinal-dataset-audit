"""Exploratory data analysis of ODIR-5K itself.

Descriptive analytics, not a new experiment: characterises the dataset
(demographics, disease prevalence, laterality, image properties) using the
same real data this project's pipeline runs on. A script rather than a
literal .ipynb on purpose -- this environment has no Jupyter kernel to
execute a notebook and capture real output reliably, and a self-contained
HTML report (same base64-inlining approach as report.py) is more portable
to view than a notebook file anyway: open it in any browser, nothing to
install. Every section function here returns (base64_png, takeaway_str)
so report.py can reuse them directly instead of duplicating the analysis.

Run: python notebooks/eda.py
Output: artifacts/eda_report.html
"""

from __future__ import annotations

import base64
import io
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

from retinaprep.adapters.odir5k import _derive_eye_column  # noqa: E402

# --- keyword -> ODIR category mapping, empirically validated against the
# real patient-level one-hot flags (see docs/notes.md): D/G/C/A/H/M match
# 99.5-100% in both directions; O matches 93.7%/99.7%; N's 58.8% reverse
# match is expected, not an error (N is patient-level, a single eye's own
# keyword can say "normal fundus" while the PATIENT is flagged abnormal
# because the fellow eye isn't normal).
CATEGORY_PHRASES = {
    "N": ["normal fundus"],
    "D": [
        "diabetic retinopathy", "nonproliferative retinopathy", "non proliferative retinopathy",
        "proliferative diabetic retinopathy", "suspected diabetic retinopathy",
        "suspicious diabetic retinopathy", "intraretinal microvascular abnormality",
        "suspected microvascular anomalies", "diabetic maculopathy",
    ],
    "G": ["glaucoma"],
    "C": ["cataract"],
    "A": ["age-related macular degeneration"],
    "H": ["hypertensive retinopathy", "arteriosclerosis"],
    "M": ["myopia", "myopic maculopathy", "pathological myopia"],
    "O": [
        "epiretinal membrane", "macular hole", "retinal detachment", "chorioretinal atrophy",
        "retinal atrophy", "branch retinal artery occlusion", "branch retinal vein occlusion",
        "central retinal artery occlusion", "central retinal vein occlusion", "optic disc edema",
        "optic nerve atrophy", "optic discitis", "myelinated nerve fibers", "choroidal nevus",
        "coloboma", "silicone oil eye", "retinitis pigmentosa", "central serous chorioretinopathy",
        "choroidal neovascularization", "pigment epithelium proliferation",
        "retinal pigment epithelial hypertrophy", "retinal pigment epithelium atrophy",
        "retinal pigmentation", "pigmentation disorder", "depigmentation", "abnormal pigment",
        "tessellated fundus", "vitreous degeneration", "vitreous opacity", "laser spot",
        "laser photocoagulation", "chorioretinopathy", "choroiditis", "spotted membranous change",
        "wedge white line change", "oval yellow-white atrophy", "glial remnants",
        "morning glory syndrome", "atrophic change", "atrophy", "peripapillary atrophy",
        "maculopathy", "retina fold", "vessel tortuosity", "retinal vascular sheathing",
        "white vessel", "drusen", "intraretinal hemorrhage", "post retinal laser surgery",
        "abnormal color of", "macular epimacular membrane",
    ],
}
CATEGORIES = ["N", "D", "G", "C", "A", "H", "M", "O"]
CATEGORY_NAMES = {
    "N": "Normal", "D": "Diabetic retinopathy", "G": "Glaucoma", "C": "Cataract",
    "A": "AMD", "H": "Hypertension", "M": "Myopia", "O": "Other",
}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def categories_in(text) -> set[str]:
    if not isinstance(text, str):
        return set()
    t = text.lower()
    return {cat for cat, phrases in CATEGORY_PHRASES.items() if any(p in t for p in phrases)}


def load_patient_df() -> pd.DataFrame:
    """One row per patient: demographics + one-hot flags. Patient-level
    columns only (age/sex/N..O are duplicated per eye in the raw file)."""
    raw = pd.read_csv(REPO_ROOT / "data" / "odir5k" / "full_df.csv")
    patients = raw.drop_duplicates(subset="ID").copy()
    patients["patient_id"] = patients["ID"].astype(str)
    return patients


def load_long_df() -> pd.DataFrame:
    """One row per eye, with `eye` derived the same way the real adapter does."""
    raw = pd.read_csv(REPO_ROOT / "data" / "odir5k" / "full_df.csv")
    raw["patient_id"] = raw["ID"].astype(str)
    raw["eye"] = _derive_eye_column(raw["filename"])
    raw["own_keywords"] = np.where(
        raw["eye"] == "L", raw["Left-Diagnostic Keywords"], raw["Right-Diagnostic Keywords"]
    )
    raw["own_categories"] = raw["own_keywords"].apply(categories_in)
    return raw


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------


def fig_age_distribution(patients: pd.DataFrame) -> tuple[str, str]:
    ages = patients["Patient Age"].dropna()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(ages, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Age")
    ax.set_ylabel("Patients")
    n_missing = patients["Patient Age"].isna().sum()
    ax.set_title(f"Age distribution (n={len(ages)}, {n_missing} missing)")
    ax.axvline(ages.median(), color="#C44E52", linestyle="--", label=f"median={ages.median():.0f}")
    ax.legend()
    fig.tight_layout()
    takeaway = (
        f"Median age {ages.median():.0f}, IQR [{ages.quantile(.25):.0f}, {ages.quantile(.75):.0f}] "
        f"— this is an older, disease-screening population, not a general-population sample; "
        f"any model trained here should not be assumed to generalise to a younger cohort."
    )
    return _fig_to_base64(fig), takeaway


def fig_sex_breakdown(patients: pd.DataFrame) -> tuple[str, str]:
    counts = patients["Patient Sex"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Patients")
    ax.set_title("Sex breakdown")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center")
    fig.tight_layout()
    frac = counts / counts.sum()
    takeaway = (
        f"{frac.idxmax()} patients are the majority ({frac.max()*100:.1f}%) — "
        f"close enough to balanced ({frac.min()*100:.1f}%/{frac.max()*100:.1f}%) that sex is "
        f"unlikely to be a dominant confound on its own, but not close enough to ignore."
    )
    return _fig_to_base64(fig), takeaway


def fig_age_by_diagnosis(patients: pd.DataFrame) -> tuple[str, str]:
    data = []
    labels = []
    for cat in CATEGORIES:
        ages = patients.loc[patients[cat] == 1, "Patient Age"].dropna()
        if len(ages) > 0:
            data.append(ages)
            labels.append(f"{cat}\n(n={len(ages)})")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Age")
    ax.set_title("Age by diagnosis category (patient may carry multiple)")
    fig.tight_layout()

    medians = {cat: patients.loc[patients[cat] == 1, "Patient Age"].median() for cat in CATEGORIES}
    oldest = max(medians, key=medians.get)
    youngest = min(medians, key=medians.get)
    takeaway = (
        f"{CATEGORY_NAMES[oldest]} patients are oldest (median {medians[oldest]:.0f}), "
        f"{CATEGORY_NAMES[youngest]} youngest (median {medians[youngest]:.0f}) — "
        f"ages track known clinical patterns (cataract/AMD skew older, myopia skews younger), "
        f"which is a sanity check that these labels behave the way real diagnoses should."
    )
    return _fig_to_base64(fig), takeaway


# ---------------------------------------------------------------------------
# Disease prevalence and comorbidity
# ---------------------------------------------------------------------------


def fig_disease_prevalence(patients: pd.DataFrame) -> tuple[str, str]:
    counts = {cat: int(patients[cat].sum()) for cat in CATEGORIES}
    order = sorted(counts, key=counts.get, reverse=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar([CATEGORY_NAMES[c] for c in order], [counts[c] for c in order], color="#4C72B0")
    ax.set_ylabel("Patients")
    ax.set_title(f"Disease prevalence (n={len(patients)} patients, not mutually exclusive)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    most, least = order[1], order[-1]  # order[0] is usually N
    ratio = counts[most] // max(counts[least], 1)
    takeaway = (
        f"{CATEGORY_NAMES[most]} is the most common pathology ({counts[most]} patients), "
        f"{CATEGORY_NAMES[least]} the rarest ({counts[least]}) — a >{ratio}x "
        f"spread between disease categories that a multi-class or per-disease task "
        f"(assessed but not run this project) would need to handle explicitly, e.g. "
        f"with class weighting, or it will under-serve the rare categories."
    )
    return _fig_to_base64(fig), takeaway


def fig_comorbidity(patients: pd.DataFrame) -> tuple[str, str]:
    disease_cats = [c for c in CATEGORIES if c != "N"]
    n_diseases = patients[disease_cats].sum(axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    counts = n_diseases.value_counts().sort_index()
    axes[0].bar(counts.index.astype(int), counts.values, color="#4C72B0")
    axes[0].set_xlabel("Number of disease categories carried")
    axes[0].set_ylabel("Patients")
    axes[0].set_title("Comorbidity count")

    comorbid = patients[n_diseases >= 2]
    pair_counts = {}
    for _, row in comorbid.iterrows():
        present = [c for c in disease_cats if row[c] == 1]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                key = tuple(sorted((present[i], present[j])))
                pair_counts[key] = pair_counts.get(key, 0) + 1
    top_pairs = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:6]
    axes[1].barh(
        [f"{CATEGORY_NAMES[a]}+{CATEGORY_NAMES[b]}" for (a, b), _ in top_pairs][::-1],
        [v for _, v in top_pairs][::-1],
        color="#DD8452",
    )
    axes[1].set_xlabel("Patients")
    axes[1].set_title("Top co-occurring pairs")
    fig.tight_layout()

    n_comorbid = int((n_diseases >= 2).sum())
    top_pair_name = f"{CATEGORY_NAMES[top_pairs[0][0][0]]}+{CATEGORY_NAMES[top_pairs[0][0][1]]}"
    takeaway = (
        f"{n_comorbid} patients ({n_comorbid/len(patients)*100:.1f}%) carry 2+ disease categories "
        f"simultaneously — comorbidity is common enough that a binary normal/abnormal task, "
        f"as used in this project's main experiment, is collapsing real clinical heterogeneity. "
        f"{top_pair_name} is the most common specific pairing ({top_pairs[0][1]} patients)."
    )
    return _fig_to_base64(fig), takeaway


# ---------------------------------------------------------------------------
# Laterality
# ---------------------------------------------------------------------------


def fig_laterality(long_df: pd.DataFrame) -> tuple[str, str, dict]:
    """For each disease category, among two-eye patients whose PATIENT-level
    flag is 1, what fraction show that category's keyword in both eyes
    (bilateral) vs only one (unilateral)?"""
    row_counts = long_df.groupby("patient_id").size()
    two_eye_ids = set(row_counts[row_counts == 2].index)
    two_eye = long_df[long_df["patient_id"].isin(two_eye_ids)]

    results = {}
    for cat in [c for c in CATEGORIES if c != "N"]:
        flagged_patients = two_eye.loc[two_eye[cat] == 1, "patient_id"].unique()
        if len(flagged_patients) == 0:
            continue
        sub = two_eye[two_eye["patient_id"].isin(flagged_patients)]
        by_patient = sub.groupby("patient_id")["own_categories"].apply(
            lambda cats_series, c=cat: sum(c in cats for cats in cats_series)
        )
        n_bilateral = int((by_patient == 2).sum())
        n_unilateral = int((by_patient == 1).sum())
        n_neither = int((by_patient == 0).sum())  # flag=1 but keyword absent both eyes
        total = len(by_patient)
        results[cat] = {
            "n": total, "bilateral": n_bilateral, "unilateral": n_unilateral, "neither": n_neither,
            "pct_bilateral_of_matched": n_bilateral / max(n_bilateral + n_unilateral, 1) * 100,
        }

    order = sorted(results, key=lambda c: -results[c]["pct_bilateral_of_matched"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bilateral_pct = [results[c]["pct_bilateral_of_matched"] for c in order]
    unilateral_pct = [100 - p for p in bilateral_pct]
    x = range(len(order))
    ax.bar(x, bilateral_pct, label="bilateral", color="#55A868")
    ax.bar(x, unilateral_pct, bottom=bilateral_pct, label="unilateral", color="#C44E52")
    ax.set_xticks(list(x))
    ax.set_xticklabels([CATEGORY_NAMES[c] for c in order], rotation=30, ha="right")
    ax.set_ylabel("% of two-eye patients with this diagnosis")
    ax.set_title("Laterality by disease category (two-eye patients only)")
    ax.legend(loc="lower right")
    fig.tight_layout()

    most_bilateral, least_bilateral = order[0], order[-1]
    takeaway = (
        f"{CATEGORY_NAMES[most_bilateral]} is the most bilateral condition "
        f"({results[most_bilateral]['pct_bilateral_of_matched']:.0f}%), "
        f"{CATEGORY_NAMES[least_bilateral]} the least "
        f"({results[least_bilateral]['pct_bilateral_of_matched']:.0f}%) — this is the direct, "
        f"per-disease breakdown behind the project's 77.8% overall concordance finding: systemic "
        f"conditions affect both eyes together far more than localised ones, so leaking a patient "
        f"leaks more label information for some diseases than others."
    )
    return _fig_to_base64(fig), takeaway, results


# ---------------------------------------------------------------------------
# Single-eye patients
# ---------------------------------------------------------------------------


def fig_single_eye_comparison(patients: pd.DataFrame, long_df: pd.DataFrame) -> tuple[str, str]:
    row_counts = long_df.groupby("patient_id").size()
    single_eye_ids = set(row_counts[row_counts == 1].index)
    two_eye_ids = set(row_counts[row_counts == 2].index)

    single = patients[patients["patient_id"].isin(single_eye_ids)]
    two = patients[patients["patient_id"].isin(two_eye_ids)]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    age_s, age_t = single["Patient Age"].dropna(), two["Patient Age"].dropna()
    age_labels = [f"two-eye\n(n={len(age_t)})", f"single-eye\n(n={len(age_s)})"]
    axes[0].boxplot([age_t, age_s], tick_labels=age_labels, showfliers=False)
    axes[0].set_ylabel("Age")
    axes[0].set_title("Age")
    t_stat, age_p = stats.ttest_ind(age_s, age_t, equal_var=False)

    sex_s = single["Patient Sex"].value_counts(normalize=True)
    sex_t = two["Patient Sex"].value_counts(normalize=True)
    idx = sorted(set(sex_s.index) | set(sex_t.index))
    x = np.arange(len(idx))
    sex_t_pct = [sex_t.get(i, 0) * 100 for i in idx]
    sex_s_pct = [sex_s.get(i, 0) * 100 for i in idx]
    axes[1].bar(x - 0.2, sex_t_pct, width=0.4, label="two-eye", color="#4C72B0")
    axes[1].bar(x + 0.2, sex_s_pct, width=0.4, label="single-eye", color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(idx)
    axes[1].set_ylabel("% of group")
    axes[1].set_title("Sex")
    axes[1].legend()
    contingency_sex = pd.crosstab(
        patients["patient_id"].isin(single_eye_ids), patients["Patient Sex"]
    )
    chi2_sex, sex_p, _, _ = stats.chi2_contingency(contingency_sex)

    abnormal_s = (single[[c for c in CATEGORIES if c != "N"]].sum(axis=1) > 0).mean()
    abnormal_t = (two[[c for c in CATEGORIES if c != "N"]].sum(axis=1) > 0).mean()
    dx_colors = ["#4C72B0", "#DD8452"]
    axes[2].bar(["two-eye", "single-eye"], [abnormal_t * 100, abnormal_s * 100], color=dx_colors)
    axes[2].set_ylabel("% with >=1 diagnosis")
    axes[2].set_title("Any abnormal diagnosis")
    contingency_dx = pd.crosstab(
        patients["patient_id"].isin(single_eye_ids),
        patients[[c for c in CATEGORIES if c != "N"]].sum(axis=1) > 0,
    )
    chi2_dx, dx_p, _, _ = stats.chi2_contingency(contingency_dx)

    fig.tight_layout()

    findings = []
    findings.append(f"age p={age_p:.4f}")
    findings.append(f"sex p={sex_p:.4f}")
    findings.append(f"diagnosis-rate p={dx_p:.4f}")
    checks = [("age", age_p), ("sex", sex_p), ("diagnosis rate", dx_p)]
    significant = [f for f, p in checks if p < 0.05]
    if significant:
        direction = "higher" if abnormal_s > abnormal_t else "lower"
        sig_list = ", ".join(significant)
        takeaway = (
            f"Single-eye patients differ significantly from two-eye patients on: {sig_list} "
            f"({', '.join(findings)}). Diagnosis rate is {direction} for single-eye patients "
            f"({abnormal_s*100:.1f}% vs {abnormal_t*100:.1f}%) — the missing eye is not random, "
            f"which is a real selection bias worth naming: whatever caused that eye to be "
            f"unavailable correlates with who these 324 patients are."
        )
    else:
        takeaway = (
            f"No significant difference found on age, sex, or diagnosis rate "
            f"({', '.join(findings)}) "
            f"— on the evidence checked here, the 324 single-eye patients look like a "
            f"random subset of the population, not a systematically different one. "
            f"Absence of evidence at n=324 "
            f"split this way is not strong evidence of absence, but nothing here suggests bias."
        )
    return _fig_to_base64(fig), takeaway


# ---------------------------------------------------------------------------
# Image properties
# ---------------------------------------------------------------------------


def fig_raw_resolution(sample_size: int = 300) -> tuple[str, str]:
    raw_dir = REPO_ROOT / "data" / "odir5k" / "ODIR-5K" / "ODIR-5K" / "Training Images"
    files = sorted(raw_dir.iterdir())
    rng = np.random.default_rng(0)
    sample = rng.choice(files, size=min(sample_size, len(files)), replace=False)
    sizes = []
    for f in sample:
        try:
            with Image.open(f) as im:
                sizes.append(im.size)
        except OSError:
            continue
    widths, heights = zip(*sizes, strict=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(widths, heights, alpha=0.4, s=15, color="#4C72B0")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title(f"Raw Training Images/ resolution (n={len(sizes)} sampled)")
    fig.tight_layout()

    n_distinct = len(set(sizes))
    takeaway = (
        f"{n_distinct} distinct resolutions in a {len(sizes)}-image sample, ranging "
        f"{min(widths)}-{max(widths)}px wide — a real mixed-camera dataset. This is exactly why "
        f"quality.py resizes to a fixed size before blur metrics (moot for the pipeline's actual "
        f"input, `preprocessed_images/`, which is uniformly 512x512, but live for this folder)."
    )
    return _fig_to_base64(fig), takeaway


def fig_quality_by_diagnosis(patients: pd.DataFrame) -> tuple[str, str]:
    quality_path = REPO_ROOT / "artifacts" / "quality.parquet"
    quality = pd.read_parquet(quality_path)
    merged = quality.merge(patients[["patient_id"] + CATEGORIES], on="patient_id", how="left")

    data, labels = [], []
    for cat in CATEGORIES:
        scores = merged.loc[merged[cat] == 1, "score"].dropna()
        if len(scores) > 0:
            data.append(scores)
            labels.append(f"{cat}\n(n={len(scores)})")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Gradability score")
    ax.set_title("Quality score by diagnosis category")
    fig.tight_layout()

    medians = {cat: merged.loc[merged[cat] == 1, "score"].median() for cat in CATEGORIES}
    cataract_median = medians.get("C")
    normal_median = medians.get("N")
    lowest = min(medians, key=medians.get)
    has_both = cataract_median is not None and normal_median is not None
    cataract_lower = has_both and cataract_median < normal_median
    if cataract_lower:
        cataract_note = (
            f"Cataract images do score lower (median {cataract_median:.3f} vs "
            f"{normal_median:.3f} for normal) — the mechanistically expected direction "
            f"(cataract causes media opacity), even though abnormal-vs-normal *overall* "
            f"showed no such gap (docs/notes.md). "
        )
    else:
        cataract_note = ""
    takeaway = (
        f"{cataract_note}{CATEGORY_NAMES[lowest]} has the lowest median quality score "
        f"({medians[lowest]:.3f}) of any category — worth checking specifically before assuming "
        f"quality curation is disease-blind at the category level, even though it looked that way "
        f"for the coarse binary label."
    )
    return _fig_to_base64(fig), takeaway


def fig_class_imbalance(manifest: pd.DataFrame) -> tuple[str, str]:
    balance = manifest["label"].value_counts(normalize=True).sort_index()
    pcts = [balance.get(0, 0) * 100, balance.get(1, 0) * 100]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["Normal (0)", "Abnormal (1)"], pcts, color=["#55A868", "#C44E52"])
    ax.set_ylabel("% of dataset")
    ax.set_title("Class balance (keywords label strategy)")
    for i, v in enumerate(pcts):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    takeaway = (
        f"{balance.max()*100:.0f}/{balance.min()*100:.0f} split is mild, not severe, imbalance -- "
        f"accuracy alone would still be a misleading metric, which is why this project reports "
        f"AUROC/AUPRC/sensitivity-at-specificity rather than accuracy, and why the train/val/test "
        f"split is stratified on label throughout."
    )
    return _fig_to_base64(fig), takeaway


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

SECTION_TEMPLATE = """
<h3>{title}</h3>
<img src="data:image/png;base64,{img}">
<p class="takeaway"><strong>Takeaway:</strong> {takeaway}</p>
"""

PAGE_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>ODIR-5K EDA</title>
<style>
body {
  font-family: -apple-system, Segoe UI, sans-serif; max-width: 1000px;
  margin: 2rem auto; padding: 0 1rem; color: #222;
}
h1, h2 { border-bottom: 2px solid #eee; padding-bottom: 0.3rem; }
h3 { margin-top: 2rem; }
img { max-width: 100%; }
.takeaway {
  background: #f5f5f5; border-left: 4px solid #4C72B0; padding: 0.6rem 1rem;
  margin: 0.5rem 0 1.5rem;
}
.note { color: #666; font-style: italic; }
</style></head><body>
<h1>ODIR-5K: Exploratory Data Analysis</h1>
<p class="note">
  Descriptive analytics on the real dataset (6392 images, 3358 patients).
  Every figure has a takeaway underneath it -- read those, not just the charts.
</p>
"""

PAGE_TAIL = "</body></html>"


def build_sections() -> list[tuple[str, str, str]]:
    """Returns (section_title, base64_png, takeaway) for every figure."""
    patients = load_patient_df()
    long_df = load_long_df()
    manifest = pd.read_parquet(REPO_ROOT / "artifacts" / "manifest.parquet")

    sections = []
    for title, fn, args in [
        ("Age distribution", fig_age_distribution, (patients,)),
        ("Sex breakdown", fig_sex_breakdown, (patients,)),
        ("Age by diagnosis category", fig_age_by_diagnosis, (patients,)),
        ("Disease prevalence", fig_disease_prevalence, (patients,)),
        ("Comorbidity", fig_comorbidity, (patients,)),
        ("Single-eye vs two-eye patients", fig_single_eye_comparison, (patients, long_df)),
        ("Raw image resolution", fig_raw_resolution, ()),
        ("Quality score by diagnosis category", fig_quality_by_diagnosis, (patients,)),
        ("Class imbalance", fig_class_imbalance, (manifest,)),
    ]:
        img, takeaway = fn(*args)
        sections.append((title, img, takeaway))

    img, takeaway, _ = fig_laterality(long_df)
    sections.insert(5, ("Laterality by disease category", img, takeaway))

    return sections


def main() -> None:
    sections = build_sections()
    html = [PAGE_HEAD]
    for title, img, takeaway in sections:
        html.append(SECTION_TEMPLATE.format(title=title, img=img, takeaway=takeaway))
    html.append(PAGE_TAIL)

    out_path = REPO_ROOT / "artifacts" / "eda_report.html"
    out_path.write_text("".join(html), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
