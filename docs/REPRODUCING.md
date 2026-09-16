# Reproducing every number in this project

This project's argument is that undocumented data handling makes reported
numbers uninterpretable. It would be a poor advertisement for that
argument if its own numbers could not be traced, and during the build
**five separate figures leaked into the documentation without a
reproducible source** — a stale 42.3% patient-overlap count, a duplicate
count corrected from 2 to 8 to 11, a scaling exponent asserted from CPU
time rather than measured, a first training run whose per-seed data was
overwritten, and a cross-patient duplicate row that compared patient
pairs against image pairs. Each was caught, but only by looking again.

So this file exists to make the distinction explicit and permanent:
**which numbers you can regenerate yourself, and which you are trusting
this document for.** A number nobody can check is not more trustworthy
for having a precise-looking decimal.

Three tiers, and every headline figure in README.md, WALKTHROUGH.md and
docs/summary.md is in exactly one of them.

---

## Before anything: what you need on disk

Nothing in `artifacts/` is committed (`.gitignore` excludes it), and no
images are in this repository. Regenerating tier 1 requires the datasets.

| Dataset | Needed for | How |
|---|---|---|
| ODIR-5K | levels 1–3, arms A–E, quality, dedupe | `bash scripts/download_data.sh` (Kaggle token required) |
| EyePACS | external validation, the 40x duplicate finding | `scripts/download_eyepacs_part.sh`, then `scripts/preprocess_eyepacs.py`; ~32.6GB for the labelled train split alone |
| IDRiD (segmentation subset) | optic disc model, mask QC, the exudate finding | Kaggle; 81 images with disc masks, plus lesion masks |
| DRIVE | vessel model, inter-grader ceiling | **see the provenance caveat in WALKTHROUGH.md §13** — the official distribution withholds the test annotations this needs |
| CHASE_DB1 | the second inter-grader ceiling | ships both observers in its normal distribution |

Paths are config-driven (`configs/default.yaml`); the segmentation
notebooks read `D:/retinaprep_data/...` constants that you will need to
point at your own copies. Every command below assumes the checked-in
config defaults, which **are** the config every reported number was
measured under — that was not always true, and fixing it is recorded in
WALKTHROUGH.md §9.

Commands are written for the venv's interpreter by explicit path, as the
rest of the project is:

```powershell
.\.venv\Scripts\python.exe -m retinaprep ingest
```

---

## Tier 1 — artifact-backed: run the command, get the number

These regenerate from disk. Where a figure is a count rather than a
statistic it is byte-reproducible; where it involves training it is
reproducible to seed-level noise, and the seed sequence is fixed.

### Levels 1–3 and the classification arms

| Number | Where it appears | Regenerate with |
|---|---|---|
| **42.0% patient overlap (1412/3358)** | README level 1, WT §1/§4 | `retinaprep ingest && retinaprep split` — exact, verified byte-identical across two clean runs |
| 0 patients crossing under `patient_group` | same | same |
| **77.8% fellow-eye concordance** (50.5% chance floor, n=3034) | README level 1, WT §5 | same, or `notebooks/findings_charts.py` |
| 45.0% normal under `keywords`; 32.9% under `normal_column`; 12.1% disagreement | data card, WT §2 | `retinaprep ingest`, then the same with `--set label.strategy=normal_column` |
| Quality reject rate 0.7% (43/6392) | arms C/D, data card | `retinaprep quality` → `artifacts/rejects.csv` |
| **13,733 phash candidates** | README "phash alone is not evidence" | `retinaprep dedupe` → `artifacts/dedupe_investigation_pairs.json` |
| **11 verified duplicate pairs, all cross-patient, over 8 patient pairs** | README level 3, WT §7 | `retinaprep dedupe` → `artifacts/duplicates.parquet` (11 clusters; group by `cluster_id` and count distinct `patient_id` pairs) |
| 3 straddling `image_random`, 2 straddling `patient_group` | README level 3 | `artifacts/duplicates_cross_split.json` |
| Arms A–D post-fix (A 0.8098, B 0.7915, C 0.7944, D 0.7856) | README split-then-curate, WT §9 | `retinaprep experiment --arm A` (…B, C, D) → `artifacts/runs/index.json` |
| **C-vs-B −0.0146 → +0.0029** | the headline self-correction | the post-fix run above; the pre-fix number needs `--set experiment.split_mode=recompute_per_arm` |
| Arms A–D pre-fix (A 0.8014, B 0.7936, C 0.7790, D 0.7820) | README "All four arms" | `retinaprep experiment --arm <X> --set experiment.split_mode=recompute_per_arm` |
| **Arm E, AUROC −0.0557 vs B** | README level 2, WT §9 | `retinaprep experiment --arm E` against the same cohort's B |

### The site audit

| Number | Regenerate with |
|---|---|
| **Site classifier 0.8397 vs 0.2932 baseline**; 97 resolutions → 43 clusters → 20 classes; χ²=115.18, p=8.8e-16; per-category breakdown incl. AMD p=0.12 | `python notebooks/domain_shift_audit.py` → `artifacts/domain_shift_audit.json` |
| EyePACS **0.9317 vs 0.2958**, χ²=139.6 | `python notebooks/domain_shift_audit_eyepacs.py` |
| Prevalence-matched band **−0.0514 to −0.0546** (0.0032 wide) | `python notebooks/arm_e_robustness_checks.py` → `artifacts/arm_e_robustness_checks.json` |
| Leave-one-site-out: −0.0529 / −0.0334 / −0.0368 / −0.0339 / **+0.0208** | same artifact, `part2_leave_one_site_out` |
| Hanley-McNeil CIs, all five spanning zero; embedding-distance check (r=+0.40, p=0.50) | `python notebooks/site4_followup.py` → `artifacts/site4_followup.json` |

### EyePACS

| Number | Regenerate with |
|---|---|
| **444 verified pairs, 441 cross-patient, 118 clusters, 68 (58%) straddling `patient_group`** | `retinaprep dedupe` against the 6,392 patient-grouped subsample → `artifacts/eyepacs_dedupe6392/` |
| Reject rate 16.9%; 99.6% tripping `fov_clipped` | `retinaprep quality` on the EyePACS manifest |
| **0.7% → 7.4% under an identical LANCZOS path** (the 24x-that-was-2.3x) | `python notebooks/quality_threshold_transfer.py` |
| Full-scan candidate count **1,005,485** (n^2.03) | `retinaprep dedupe` on all 35,126 — ~9 CPU-hours. Its *cluster* output is deliberately unused; see WT §12 |

### The segmentation strand

| Number | Regenerate with |
|---|---|
| **Ceilings: DRIVE 0.7879 ± 0.0206 (n=20), CHASE_DB1 0.7765 ± 0.0250 (n=28)**, at native resolution | `python notebooks/inter_grader.py` → `artifacts/inter_grader.json` |
| **Model vs both observers: 0.7884 / 0.8074 / ceiling 0.7882**, style gap −0.0190, 13/20 | `python notebooks/vessel_vs_ceiling.py` → `artifacts/drive_vessel/vessel_vs_ceiling.json` |
| Enrichment **1.57x** [1.50, 1.63]; side-taking **0.4452** [0.4280, 0.4624] | `python notebooks/disagreement_location.py` |
| Disc Dice **0.8581 ± 0.1646**, median 0.9110 (n=27); vessel 0.7884 (16/4/20 split) | `artifacts/idrid_od/unet_od_result.json`, `artifacts/drive_vessel/unet_vessel_result.json` |
| **Exudate finding**: gradability 0.776 vs 0.708, 0 of 3 flagged, **5.51x** burden, ranks #1/#2, p=0.0595, per-image geometry | `python notebooks/idrid_failure_analysis.py` → `artifacts/idrid_od/failure_analysis.{csv,json}` |
| Mask QC: 81 masks, 0 rejected, 1 area-outlier flag | `artifacts/idrid_od/mask_quality_summary.json` |

> **Why DRIVE's ceiling has two values.** `inter_grader.py` compares masks
> at native 565×584 (**0.7879**); `vessel_vs_ceiling.py` recomputes on the
> 512×512 grid the model predicts on (**0.7882**), so that the model and
> the humans are scored identically. Same quantity, two grids, 0.0003
> apart. Both are reported rather than one being quietly chosen.

### Figures and reports

```powershell
.\.venv\Scripts\python.exe notebooks\eda.py              # docs/eda_report.html
.\.venv\Scripts\python.exe notebooks\findings_charts.py  # docs/findings_report.html
.\.venv\Scripts\python.exe -m retinaprep report          # docs/qc_report.html
.\.venv\Scripts\python.exe notebooks\readme_figures.py   # all 8 docs/figures/*.png
```

---

## Tier 2 — prose-sourced: the inputs no longer exist

These were real measurements. Their inputs were not persisted, so the
number is only as good as this project's record of it. **They are cited
from the documents, not regenerated**, and none of them is load-bearing
for the central claim.

| Number | Where | Why it cannot be regenerated |
|---|---|---|
| **Run 1 A-vs-B: +0.0173, p=0.037, 5/5 seeds** | README level 1, WT §1/§9 | Per-seed JSON was **overwritten** by a later run using the same `<arm>_seed<n>` paths. The aggregate survived in the docs; the raw data did not. This is what motivated the unique-run-directory fix (WT §9), so the failure is recorded rather than hidden. The *second* run (+0.0079) is tier 1. |
| Run 1 variance figures (A 0.0069 vs B 0.0083; Sens 2.58x) | WT "Is the naive split unstable" | same overwrite |
| **Patient-consistency fix: 10 images reassigned, 0 patients inconsistent, overlap 0/0/0, train 4008→4006, val 402→404** | README level 2 robustness, WT §9 | `enforce_patient_site_consistency` exists in `notebooks/domain_shift_audit.py` and re-running reproduces the reassignment, but no JSON records these five numbers. Closest thing to a tier-1 promotion available cheaply. |
| Two hazy images scoring 0.943 and 0.979 | data card blind spot | one-off inspection over a since-regenerated quality table; the *blind spot* is reproducible, the two specific scores are cited |
| Raw `Training Images/` reject rate 4.2% vs 0.7% | WT interview Q3 | ad-hoc run against a raw folder outside the pipeline |
| 69.4% overlap from removing 43 *random* images | README level 1 hypothesis 3, WT §9 | one-off control; the 69.2% real figure it is compared against is tier 1 |
| Visual confirmations — EyePACS difference maps, ODIR-5K phash false positives, the 10 sampled label disagreements, the three rendered disc failures | throughout | judgements by eye, recorded as such. `artifacts/idrid_od/failures.png` and `artifacts/eyepacs_dedupe6392/*.png` persist the images; the conclusions drawn from them are not numbers |

---

## Tier 3 — stated as unmeasured

Listed so that nobody mistakes an absence for a finding.

| Claim | Status |
|---|---|
| EyePACS's **dataset-wide** duplicate count | **Unknown.** Higher than 444, does not scale linearly. The full scan ran, but transitive chaining corrupts its clusters (WT §12), so its totals — 16,782 pairs, 5,721 images, 492 clusters — are **not used anywhere** and should not be quoted |
| Whether any published result was affected by EyePACS duplication | **Not shown, and not shown deliberately.** Establishing it would require re-running each study with its own split |
| "Site" as ground truth | A **resolution-derived proxy** in both datasets. It under-counts wherever two cameras share a resolution, so every site count is a lower bound |
| Whether the ~0.78 ceiling generalises beyond DRIVE and CHASE_DB1 | **n=2.** WT §14 states the falsification condition: a third two-observer vessel dataset near 0.78 strengthens it, one at 0.85 refutes it |
| Whether a recall-weighted loss flips side-taking above 0.5 | **Not run.** The prediction is written down in WT §16 before anyone tests it |
| Per-disease (rather than binary) leakage sensitivity | Assessed as plausible, never measured (WT §5) |

---

## The one-command check

Everything that needs no dataset at all:

```powershell
.\.venv\Scripts\python.exe -m pytest      # 52 tests, synthetic fixtures only
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\doctor.py
```

If those pass on a clean checkout, the code is intact; everything above
is about whether the *numbers* are.
