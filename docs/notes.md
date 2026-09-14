# Working notes

Scratch space. Record decisions here as they are made so WALKTHROUGH.md can be
written from evidence rather than reconstructed from memory.

## Log

- [ ] Session 1: scaffold created, ingest + split + A/B experiment.

## Provenance: which numbers can be regenerated from disk, and which can't

Prompted by finding a stale headline number (the 42.3%/1422 patient-overlap
figure, corrected above/below and in README.md) that had silently drifted
from what `artifacts/` actually reproduces. Being explicit about which
numbers in this project are artifact-sourced (regenerate them yourself,
right now, with the command given) versus prose-sourced (a one-off
investigation whose inputs are gone or were never saved) is itself part of
the honesty this project is arguing for -- a number nobody can check is
not more trustworthy for having a precise-looking decimal.

**Artifact-sourced (regenerate yourself):**

- Patient overlap (42.0%/1412/3358) and fellow-eye concordance (77.8% vs
  50.5%) -- `python -m retinaprep ingest && python -m retinaprep split`
  from a clean `artifacts/`, or `notebooks/findings_charts.py`.
- The current 4-arm results table (post split-then-curate fix,
  `split_mode: persisted_base`, natural sizes) -- `artifacts/runs/index.json`
  and `artifacts/runs/*/metrics.json`, or `python -m retinaprep
  experiment --arm <A|B|C|D>` to add a fresh cohort (see the run-index
  note below).
  The *pre-fix* 4-arm table ("Arms C and D: full 4-arm run" below,
  `split_mode: recompute_per_arm`, capped to 4435) is also still
  reproducible, via `python -m retinaprep experiment --arm <A|B|C|D>
  --set experiment.split_mode=recompute_per_arm` -- unlike the truly-lost
  numbers below, this one only needs an explicit override, not
  archaeology.
- Duplicate cluster count/cosine-similarity separation -- `python -m
  retinaprep dedupe`, or `notebooks/findings_charts.py`'s live ResNet18
  embedding pass.
- Quality reject rate and rejects.csv reasons -- `python -m retinaprep quality`.
- Every EDA figure in `notebooks/eda.py` -- computed directly from
  `full_df.csv` / `manifest.parquet` each time it runs.

**Prose-sourced (the investigation's inputs no longer exist as an
artifact; the number is only as good as this document's record of it):**

- The *first* full-scale A/B run's per-seed AUROC/AUPRC/Sens (cap=4473,
  "Full-scale A/B run" section below) -- superseded on disk by the second
  (4-arm, cap=4435) run, which reused the same run-directory names before
  the unique-run-directory fix (see "artifacts/runs/ no longer overwrites"
  below). Cited from this document's table in
  `notebooks/findings_charts.py`'s `fig_ab_replication`.
- The exploratory small-scale CPU run (subsample_n=800, epochs=4, "First
  A/B training run" section below) -- already flagged in this document as
  "not the headline number"; also not reproducible from current artifacts.
- The train-set-overlap falsification test (69.2% real vs 69.4% random
  control, "Why curation costs AUROC" (b)/(c) below) -- a report-only, ad
  hoc investigation with no export step; nothing in the pipeline persists
  arm-specific curated training pools or their overlap.
- The original 400-patient fellow-eye cosine-similarity sample (mean
  0.933, min 0.770, max 0.975) used to justify `embedding_cosine_min:
  0.99` ("dedupe.py" section below) -- a one-off sampled check, superseded
  by `findings_charts.py`'s full-population (n=3034) live recomputation,
  which is artifact-sourced and should be treated as the current number.
- The quality-vs-disease chi-square check on the 43 rejects (62.8% normal
  vs 45.0% baseline, p=0.019, "Why curation costs AUROC" (a) below) -- ad
  hoc, not behind a pipeline command.
- The per-epoch training-curve analysis (peak-then-decline pattern, "Model
  capacity" section below) -- read once off training logs, not from a
  structured artifact.

**Why this happened, and what changed so it's less likely to recur:**
`configs/default.yaml` used to default to `dataset.subsample_n: 2000` and
`experiment.n_seeds: 1` -- neither matches any reported headline number,
which all used the full dataset and 5 seeds via undocumented CLI
overrides. The checked-in defaults now match what every reported number
actually used (full dataset, 5 seeds), so a clean `ingest`/`split`/
`experiment` run with no overrides reproduces the real thing rather than
a silently different small one. Separately, `train.py` used to write
every run to `artifacts/runs/<arm>_seed<seed>/metrics.json` -- a fixed
name that a later run with the same arm/seed silently overwrote (this is
exactly how the first full-scale A/B run's per-seed data was lost). Each
run now gets a unique directory (`<name>_<UTC timestamp>_<8-char config
hash>`) and an entry in `artifacts/runs/index.json`; nothing overwrites,
and `retinaprep.utils.load_current_run_metrics` selects, per arm, only
the runs sharing that arm's most recently recorded config hash, so an
old and a new cohort can coexist on disk without blending into one
meaningless average.

## Split-then-curate ordering fixed: the confound diagnosis, confirmed by eliminating it

The "Why curation costs AUROC" investigation below diagnosed, but did
not fix, a real defect: each arm recomputed `patient_group_split` fresh
on its own curated pool, and `StratifiedGroupKFold` reshuffles fold
membership aggressively on any pool change -- removing 43 images (0.7%)
resampled ~31% of the training set, so B's and C's training sets shared
only 69.2% of images despite matched size (69.4% for a random-removal
control, confirming the mechanism was the split algorithm, not
curation). That made the original C-vs-B comparison measure mostly
split-algorithm sensitivity to perturbation, not curation's effect --
flagged as a design lesson for future work, not implemented at the
time.

**The fix**: split once on the raw pool (already what `retinaprep split`
persists), then for curated arms filter that fixed split down to
whichever images survive curation, instead of recomputing. Implemented
in `splits.py` (`load_persisted_split`, `filter_split_to_manifest`) and
wired into `experiment.py` via `experiment.split_mode: persisted_base`
(new default; `recompute_per_arm` kept, fully working, and selectable
for reproducing the numbers below the fold). A direct consequence,
stated as a design decision rather than left implicit: **arm sizes now
differ naturally and are not capped back to a common size**
(`match_arm_sizes` is ignored under this mode) -- matching size is
exactly what forced the fresh per-arm recompute in the old design, so
re-introducing it here would silently reintroduce the defect. Under
`persisted_base`, the size difference between arms *is* the treatment
(how much curation actually removed), not a confound to correct for.

**Verification that the fix does what it's supposed to** (no training
needed for this check -- pure split/curation logic against the real
6392-image dataset):

| | Old (recompute_per_arm) | New (persisted_base) |
|---|---|---|
| B train size | 4474 | 4474 |
| C train size | 4435 (capped) / 4443 (natural) | 4446 |
| D train size | 4435 (capped) / 4435 (natural) | 4436 |
| B/C training-set overlap | 69.2% | **100%** (C is a strict subset of B) |
| B/D training-set overlap | -- | **100%** (D is a strict subset of B) |

C's and D's training sets are now provably subsets of B's -- every
image that survives curation keeps the exact fold `retinaprep split`
originally assigned it. 28 of the 43 quality-rejected images happened
to land in B's train fold (4474 - 4446 = 28); 4 landed in val, 11 in
test. This is the number the fix was supposed to produce, and it does.

**A small, worth-stating side-confirmation**: a random draw of 43 items
across B's train/val/test folds (sizes 4474/639/1279) would expect
~30.1/4.3/8.6 -- close to the observed 28/4/11 (chi-square goodness of
fit: 0.837, p=0.658, though the val cell's expected count is below the
usual rule-of-thumb minimum of 5, so treat the p-value as indicative,
not precise). Quality rejection isn't correlated with fold membership --
a small but genuine piece of evidence that the 43 rejects aren't
clustered in a way that would itself bias which fold curation's effect
gets measured against.

**Prediction, stated before re-running anything**: with ~43 images
removed out of ~4470 and fold membership now stable, C-vs-B should show
a much smaller effect than the -0.0146 previously measured, because
most of that original effect was resampling, not curation.

**Re-ran all four arms, 5 seeds, full dataset, GPU, `split_mode:
persisted_base`, natural (uncapped) sizes.** Results:

| Arm | N train | AUROC | AUPRC | Sens@95%Spec |
|---|---|---|---|---|
| A (image_random) | 4473 | 0.8098 +/- 0.0069 | 0.8567 +/- 0.0052 | 0.4554 +/- 0.0262 |
| B (patient_group) | 4474 | 0.7915 +/- 0.0099 | 0.8439 +/- 0.0104 | 0.4327 +/- 0.0304 |
| C (quality-curated) | 4446 | 0.7944 +/- 0.0094 | 0.8462 +/- 0.0076 | 0.4020 +/- 0.0142 |
| D (quality+dedupe) | 4436 | 0.7856 +/- 0.0135 | 0.8395 +/- 0.0083 | 0.4191 +/- 0.0158 |

**The prediction held.** Same 6-test-family Bonferroni bar as the
original analysis (alpha=0.05/6=0.00833, t_bonf(df=4)=4.851):

| Comparison | Metric | Before (recompute_per_arm) | After (persisted_base) |
|---|---|---|---|
| C - B | AUROC | -0.0146, p=0.0078 (survives), sign 5/5 | **+0.0029, p=0.6160 (nowhere close), sign 3/5** |
| C - B | AUPRC | -0.0068, p=0.1025, sign 0/5 | +0.0023, p=0.6275, sign 2/5 |
| C - B | Sens@95 | +0.0041, p=0.7405, sign 3/5 | -0.0307, p=0.0384 (uncorrected only), sign **5/5** |
| D - B | AUROC | -0.0116, p=0.1403, sign 1/5 | -0.0060, p=0.5062, sign 3/5 |
| D - B | AUPRC | -0.0046, p=0.4265, sign 2/5 | -0.0044, p=0.5223, sign 2/5 |
| D - B | Sens@95 | +0.0288, p=0.1456, sign 4/5 | -0.0135, p=0.3707, sign 1/5 |

**AUROC/AUPRC C-vs-B: the one comparison in this entire project that
survived Bonferroni correction has disappeared and flipped direction
under the fix.** This is not a disappointing result -- it is the
significant result. The project diagnosed a methodological artifact
(split-algorithm sensitivity to perturbation, masquerading as a
curation effect) from indirect evidence (the 69.2%/69.4% overlap
numbers) and predicted in writing what eliminating it should do to the
headline comparison; the prediction held, on the same statistical bar
applied to every other claim here. D-vs-B stays a null under both
designs, consistent with dedupe removing too few images to plausibly
move a metric either way.

**One new, honest, not-fully-confirmed lead surfaced by the fix**:
Sens@95%Spec now shows a sign-consistent (5/5) *decrease* for C vs B
(-0.0307, p=0.0384 uncorrected) that does not survive Bonferroni
correction (0.0384 > 0.00833). Reported plainly as an uncorrected,
sign-consistent, unconfirmed signal -- not promoted to a finding, for
the same reason the earlier variance-instability lead wasn't: it hasn't
cleared the bar this project holds everything else to. Worth watching
with more seeds, not worth a headline claim at n=5.

**A-vs-B, a third measurement, with a methodological caveat**: this run
also re-measured A-vs-B (needed since the base split, and whether sizes
are capped, both changed): AUROC diff +0.0183, t=2.73, p=0.0525
(uncorrected), sign 4/5 -- between the first run's +0.0173 (p=0.037,
5/5) and the second run's +0.0079 (p=0.29, 3/5). Not a clean third trial
of the same design, stated plainly rather than glossed over: this run
used natural, uncapped sizes (A=4473, B=4474) rather than either
previous run's matched-size cap (4473 and 4435 respectively), so a
difference in training-set size is now confounded with whatever
between-run variation produced this number. Reported as a third data
point in an ongoing pattern, not as resolving it.

**Old numbers, kept, not deleted**: the pre-fix 4-arm table, all
per-seed values, and the full Bonferroni-adjusted CI analysis are in
"Arms C and D: full 4-arm run" below, produced under
`experiment.split_mode: recompute_per_arm` (still fully working,
selectable via config, and exactly reproducible) at `match_arm_sizes:
true` (capped to 4435). Both README.md and WALKTHROUGH.md now present
both sets of numbers, clearly labelled before-fix and after-fix.

## Cross-camera domain-shift audit: site is recoverable, and it's entangled with diagnosis

Roadmap item 2. ODIR-5K mixes Canon, Zeiss and Kowa across multiple
Chinese centres; there is no explicit camera/site column. Raw image
resolution (before this project's own preprocessing resizes everything
to 512x512) is used as a proxy for site. **Stated once, applies to
every number below: this is a proxy, not the camera itself.** If
several cameras happen to share a resolution, this under-counts real
sites -- everything here is a lower bound on how much site variation
exists, not an exact count.

**Deriving the proxy**: matched all 6392 manifest filenames against the
raw `Training Images/` folder (all 6392 found, confirmed directly) and
recorded each one's pre-preprocessing resolution -- 97 distinct
(width, height) pairs. Exact-tuple grouping is too fragmented to treat
as "sites" directly (many are almost certainly the same camera with a
slightly different crop), so the 97 distinct resolutions were
DBSCAN-clustered (eps=100 raw pixels, min_samples=1, chosen by
inspection: this eps merges near-identical resolutions without chaining
distant ones together) into 43 raw clusters. Clusters holding fewer
than 50 images (24 of the 43) were consolidated into one "Other" class,
since a class that small can't be meaningfully split or evaluated --
**20 final site classes**: 19 real clusters (1982 down to 53 images
each) plus "Other" (255 images, the long tail). Checked directly
(not assumed): 3003/3034 (99.0%) two-eye patients have both eyes at the
identical raw resolution, and only 10/3034 (0.3%) are assigned to
different site clusters after DBSCAN -- resolution, and therefore this
proxy, is overwhelmingly a per-patient property, not a per-image one.

**The test**: a fresh ResNet18 (new 20-class head), trained on the
PREPROCESSED images -- already resized to a common 512x512, exactly
like every other model in this project -- to predict site_label.
Patient-grouped split (train=4473/val=640/test=1279, same proportions
as the main pipeline), so a patient's fellow eye couldn't hand the
classifier a trivial shortcut (the same leakage concern this entire
project is about, applied here to a different target). 15 epochs max,
patience 3, same hyperparameters as train.py otherwise.

**Result: test accuracy 0.8397, against a majority-class baseline of
0.2932 (best val accuracy 0.8719, epoch 11 of 14 before early
stopping).** The model recovers site correctly 84% of the time on
held-out patients, nearly 3x the naive baseline. **This is the
finding the test was designed to surface**: the signal survived being
resized to a common 512x512 -- it cannot be pixel dimensions, since
every image the classifier ever saw was the same size. It has to be in
the optics, sensor response, or colour rendition. Per-class recall is
uneven (0.82-1.00 for the 12 largest classes; 0.36-0.56 for the
smallest ones and "Other", which is exactly the pattern expected from
class-size imbalance, not a red flag) and errors concentrate on specific
pairs (e.g. true=site_11 predicted=site_1, 24/50 times) rather than
scattering randomly, consistent with some DBSCAN clusters being the same
real camera split across two resolution buckets rather than genuinely
different confusions.

**Then the part that matters**: does site correlate with diagnosis?
Patient-level contingency table, site_label (20 classes) x the
patient-level `N` flag (any diagnosed abnormality) -- chosen over the
per-eye `label` column specifically because `N` is genuinely
patient-level and `label` is not (see "Split-then-curate" section
above's sibling finding: `label` disagrees across a patient's own eyes
22.2% of the time, which would make picking one eye's value for a
patient-level question arbitrary).

**chi2=115.18, p=8.8e-16, dof=19 -- site and diagnosis are entangled,
not independent.** Supplementary breakdown by individual category
(same site_label x each of N/D/G/C/A/H/M/O), because "does site
correlate with diagnosis" is more useful as "which diseases, specifically" --

| Category | chi2 | p |
|---|---|---|
| D (diabetic retinopathy) | 171.97 | 1.2e-26 |
| H (hypertensive retinopathy) | 174.79 | 3.3e-27 |
| G (glaucoma) | 100.30 | 4.7e-13 |
| C (cataract) | 82.03 | 8.3e-10 |
| O (other) | 42.93 | 1.3e-03 |
| M (myopia) | 39.34 | 4.0e-03 |
| A (AMD) | 26.36 | 0.12 (not significant) |

Every category except age-related macular degeneration shows a highly
significant site correlation -- diabetic retinopathy and glaucoma, the
two categories named as the concrete concern before running this, are
the two strongest (alongside hypertensive retinopathy, not originally
named but showing the single largest chi2 of all eight). AMD being the
one exception is itself a real, specific, checkable finding, not a gap
in the analysis -- worth a direct look in a later session rather than
waved past.

**What this means for the leakage findings in this project**: site and
disease are demonstrably entangled in this dataset, for most disease
categories, at a significance level that isn't close to marginal.
Patient-level splitting (this project's entire design) prevents a model
from memorising a specific *patient's* fellow eye across the train/test
boundary, but it does nothing to prevent a model from learning
*site-correlated* shortcuts that generalise across many different
patients captured at the same centre -- a model could score well on
disease classification partly by recognising which clinic's camera took
the photo, entirely compatibly with a patient-grouped split, since
patient-grouping was never designed to address this axis at all.
**Patient-level splitting is necessary but not sufficient; site-level
splitting (holding out entire sites, not just entire patients) is the
stricter standard this dataset would need to fully rule out a site
confound.** Not implemented in this project -- flagged as a genuinely
useful design conclusion for future work, in the same spirit as the
split-then-curate ordering fix above: diagnosed from evidence, stated
plainly, not acted on beyond the diagnosis in this pass.

Full classifier result and confusion matrix: `artifacts/domain_shift_audit.json`
(gitignored, regenerate with `python notebooks/domain_shift_audit.py`).

### Follow-up A: verifying the classifier isn't reading fold structure

Before trusting the 84% accuracy above, checked directly whether
`build_split` (grouped on `patient_id` only, not stratified on
`site_label`) could have concentrated sites by fold rather than
producing genuinely held-out representation for each one. Reproduced
the exact split deterministically (same seed, no retraining needed) and
tabulated every site class's image count in every fold.

**All 20 site classes appear in all three folds** -- no class is absent
from test. Per-class train-fraction (target ~0.70, matching the
train/val/test sizes 4473/640/1279) ranges 0.629 (site_8) to 0.825
(site_16, a small 63-image class most exposed to random group-assignment
noise); the rest sit closer to 0.65-0.73. Reasonably proportional, no
site concentrated into a single fold. **The 84% accuracy is not an
artifact of fold structure** -- every site had genuine held-out images
to be evaluated against.

### Follow-up B: arm E -- site-level splitting, the direct experiment

The audit above diagnosed patient-level splitting as necessary but
insufficient. The direct test: hold out entire sites and measure how
much the site shortcut is worth.

**`splits.site_group_split`**: groups on `site_label` (loaded from
`artifacts/site_labels.parquet`, written once by
`notebooks/domain_shift_audit.py`) instead of `patient_id`, same
`StratifiedGroupKFold` mechanics as `patient_group_split` otherwise.
Patient-level integrity comes free rather than needing separate logic:
99.0% of two-eye patients share an identical raw resolution (and
therefore site_label) across both eyes, established in the audit above
-- grouping by site overwhelmingly keeps a patient's eyes together too.
**Not a perfect guarantee, checked directly rather than assumed**:
`patient_overlap` on the persisted `site_group` split reports **6**
patients straddling train/val (vs 0 for `patient_group`) -- the
~10 patients whose eyes landed in different site clusters (Follow-up A
run's own log) can, and in 6 cases did, end up split across a site_group
fold boundary. Small, real, and reported rather than glossed over.

Wired into `experiment.py` as **arm E** (`curation: raw, split:
site_group`), using the same persisted-split discipline as the item-1
fix: `retinaprep split` computes and persists `site_group.json` once
(only if `site_labels.parquet` already exists -- skipped with a log
message otherwise, so the base two splits keep working for anyone who
hasn't run the domain-shift audit), and arm E loads it directly, no
recompute, no curation-driven filtering (E's pool is the raw manifest,
same as B).

**A caveat serious enough to state before the split sizes**: with only
20 groups and a highly skewed size distribution (site_0 alone is 31% of
the dataset), `StratifiedGroupKFold` had very little room to manoeuvre.
**The val fold is a single site** (`site_2`, 402 images) **and the test
fold is a single site** (`site_0`, 1982 images) -- not "one site
dominates a mixed fold," but the fold *is* one site. Train=4008 (62.7%,
short of the configured 70% target -- group-based splitting on 20
unevenly-sized groups doesn't hit configured fractions precisely, the
same caveat `patient_group_split` already carries, just more visible
here with far fewer groups to work with).

**Class balance shifts substantially as a direct, expected consequence
of the entanglement measured above, not a bug in the split**: train
47.3% normal, val 61.2% normal, test **37.0%** normal (63% abnormal) --
because site correlates with diagnosis, holding out a site necessarily
means holding out that site's own disease mix, which differs from the
rest of the dataset's. This makes a same-scale head-to-head comparison
against arm B's 55.0%-abnormal test set imperfect -- flagged plainly,
not smoothed over.

**Prediction, stated before running anything**: AUROC should drop
substantially relative to arm B, and that drop is the measurement of
the site shortcut. Seed-to-seed variance was also predicted to be
higher than B's, since a single dominant site in the test fold gives
the model much less genuinely varied held-out material per seed.

**Ran arm E, 5 seeds, full dataset, same hyperparameters as every other
arm.** Same 3-metric Bonferroni family as the original A-vs-B analysis
(alpha=0.05/3=0.0167, t_bonf(df=4)=3.961):

| Metric | E-B mean diff | t | p | 95% CI | Bonferroni CI | Sign | Survives |
|---|---|---|---|---|---|---|---|
| AUROC | -0.0557 | -6.70 | 0.00259 | [-0.0787,-0.0326] | [-0.0886,-0.0227] | 5/5 | **YES** |
| AUPRC | +0.0023 | +0.41 | 0.70592 | [-0.0133,+0.0179] | [-0.0200,+0.0245] | 3/5 | no |
| Sens@95%Spec | -0.0585 | -4.80 | 0.00867 | [-0.0923,-0.0246] | [-0.1068,-0.0102] | 5/5 | **YES** |

**The prediction held, decisively.** AUROC drops -0.0557 (over 3x the
original patient-level effect, +0.0173/+0.0079/+0.0183 across the
three A-vs-B runs) and survives Bonferroni correction with a
confidence interval that excludes zero even after correcting -- the
cleanest, most statistically decisive result in this entire project.
Sens@95%Spec shows the same pattern. **AUPRC does not** -- plausibly
*because of*, not despite, the class-balance shift just flagged: AUPRC's
precision baseline scales directly with test-set prevalence (63%
abnormal here vs ~55% for B), while AUROC is a rank-based measure
invariant to class-prior shift by construction. Read together, the
three metrics' pattern is informative rather than contradictory: the two
prevalence-insulated metrics (AUROC, and Sens@95%Spec at a fixed
operating point) show a large, consistent, corrected-significant drop;
the one prevalence-sensitive metric doesn't, for a specific, statable
reason rather than an unexplained inconsistency.

**Variance prediction: directionally right, not statistically
confirmed** -- checked with Levene's test rather than eyeballed, the
same discipline applied to the earlier (also-unconfirmed) naive-split
variance lead. AUROC std: E=0.0194 vs B=0.0099 (1.96x), Levene p=0.451
-- suggestive, not significant. AUPRC: 1.23x, p=0.893. Sens@95%Spec:
0.86x (the *opposite* direction from predicted), p=0.736. Reported as
exactly that: the AUROC ratio is directionally consistent with the
prediction but doesn't clear significance at n=5, and Sens@95%Spec
doesn't support it at all.

**What this confirms**: the audit's diagnosis wasn't just a statistical
association -- removing the ability to exploit it (by holding out
entire sites) costs a model substantially more than removing patient
overlap ever did. Patient-level splitting is necessary but not
sufficient; this is the direct, measured demonstration of that claim,
not just the entanglement evidence for it. **Resolution remains a proxy
for camera, not the camera itself, throughout this entire result** --
restated because it applies to arm E's site definition exactly as much
as to the audit that produced it.

### Follow-up C: is arm E's -0.0557 a prevalence artifact, or a single-site fluke?

Two confounds stood between arm E's headline number and a clean claim:
its test set's class balance differs from arm B's (63% vs 55% abnormal),
and its test fold *is* a single site (site_0) -- one observation, not a
distribution. Both checked directly rather than argued away.

**Fix first: the 6 straddling patients.** `site_group`'s own overlap
report (Follow-up B) showed 6 patients splitting across a fold boundary
where `patient_group` has 0. Verified the mechanism directly instead of
assuming it: computed raw-resolution clusters fresh and checked, per
patient, how many distinct site labels their two eyes carry. **10
patients** (not just the 6 that happened to straddle a fold -- 4 more
have inconsistent labels that both landed in the same multi-site train
fold, so didn't show up as fold-crossing) have two eyes at genuinely
different raw resolutions, confirming the guess exactly: the site proxy,
built from image resolution, is not perfectly patient-consistent because
resolution is a per-*image* property and a patient's two eyes were not
always captured with the same equipment/settings.

**Fix**: `enforce_patient_site_consistency` (in `domain_shift_audit.py`)
assigns each patient's *pair* of images to one label -- their majority
label, breaking ties by preferring the larger class over the `Other`
bucket. Reassigns exactly the minority eye for each affected patient:
**10 images changed, across 10 patients, 0 remaining inconsistent**
(checked directly after the fix, not assumed). Re-ran the full
`site_group` split on the corrected labels: patient overlap drops to
**0/0/0** across train/val/test (checked directly: train patients=2115,
test patients=1032, zero set-intersection all three ways). Effect on the
split itself is small, as expected for a 10-image change on a 6392-image
pool: train 4008->4006, val 402->404 (net 2 images moved train->val),
**test fold unchanged in count and identity (site_0, 1982 images)** --
arm E's headline test fold is not touched by this fix at all. Re-training
was still required for the checks below because model weights and
per-example predictions were never persisted in the original runs (see
`train.run_train`'s new `save_predictions` argument) -- an infrastructure
gap independent of this fix, closed while doing this work.

**Check 1 -- prevalence-matched re-evaluation (no retraining of the
*mechanism* under test, but retraining was needed to get predictions to
re-evaluate at all).** Retrained B and E, 5 seeds each (42-46), same
config, `record_run=False` so this doesn't pollute the official results
table, `save_predictions=True` to get per-example scores. Baseline gap on
this retrained cohort: **-0.05452** (AUROC, E-B, 5-seed mean) -- close to
but not identical to the originally reported -0.0557; the ~0.001-0.002
drift is retraining stochasticity at matched seeds, already characterized
elsewhere in this project (arm E's own AUROC std is 0.0194), not a
regression, and small next to the effect being measured.

Then, per test-set direction:

| Comparison | B (as-is) | E (as-is) | AUROC gap (E-B) |
|---|---|---|---|
| Original prevalences (B 55.0% abn, E 63.0% abn) | 0.79153 | 0.73701 | **-0.05452** |
| Matched at B's 55% (E's test subsampled down) | 0.79153 | 0.73695 | **-0.05457** |
| Matched at E's 63% (B's test subsampled up) | 0.78838 | 0.73701 | **-0.05137** |

All three gaps sit inside a **0.0032-wide band** (-0.0514 to -0.0546) --
under 6% of the gap's own magnitude. Subsampling B's test set up to E's
63% abnormal rate shrinks the gap only slightly (-0.0546 -> -0.0514);
subsampling E's test down to B's 55% barely moves it at all (-0.0546 in
the other direction). **This resolves the ambiguity the check was
designed to test: the AUROC drop survives prevalence-matching in both
directions, essentially unchanged. It is not a prevalence artifact.**
Consistent with AUROC's rank-based, prevalence-invariant construction --
and consistent with, without further evidence either confirming or
refuting, AUPRC's earlier non-movement (p=0.706): AUPRC's own prevalence
sensitivity was one plausible reason it didn't move, and nothing here
contradicts that reading, but nothing here proves it was *the* reason
either -- stated as exactly that level of confidence, not resolved in the
finding's favour beyond what was actually checked.

**Check 2 -- leave-one-site-out across the 4 largest named sites (`Other`
excluded: it is a merged bucket of small clusters, not a real site).**
For each of site_1 through site_4 in turn: hold out that entire site as
test, `GroupKFold` on `patient_id` for the train/val split of everything
else, train 3 seeds, compare to the same 3-seed B baseline
(mean AUROC 0.79170). site_0 (arm E's original test fold) is included
using the retrained cohort from Check 1 as a fifth point of reference.

| Held-out site | n_test | abnormal frac | mean AUROC (3 seeds) | gap vs B |
|---|---|---|---|---|
| site_0 (original arm E) | 1982 | 0.630 | 0.7388 | **-0.0529** |
| site_1 | 501 | 0.489 | 0.7583 | **-0.0334** |
| site_2 | 404 | 0.384 | 0.7549 | **-0.0368** |
| site_3 | 379 | 0.491 | 0.7578 | **-0.0339** |
| site_4 | 336 | 0.452 | 0.8125 | **+0.0208** |

**4 of 5 held-out sites replicate the direction**, at magnitudes
(-0.033 to -0.053) comparable to or only somewhat smaller than the
original arm E result -- this is not a phenomenon that only shows up for
the one site that happened to land in the original test fold. **site_4
is a genuine, unexplained exception**, and is reported as found rather
than smoothed over: +0.0208, and not a seed fluke -- all three seeds
individually score higher than the B baseline (0.7983, 0.8126, 0.8268),
a tighter spread than several of the negative-gap sites. Checked and
ruled out as an explanation: site_4's class balance (45.2% abnormal) is
unremarkable, close to site_1 (48.9%) and site_3 (49.1%), both of which
*do* show the expected negative gap -- prevalence is not what
distinguishes site_4. Noted, not resolved: site_4 has the smallest test
fold of the five (336 images, vs 379-1982 for the others), which widens
the sampling uncertainty on its point estimate purely from which 336
images happen to be in it -- a reason for humility about the exact
magnitude, not a specific causal account of why the sign flips.

Statistics across the 5 site-level observations, treated as an
independent sample (n=5, not the within-site 3-seed n): one-sample
t-test of the 5 gaps against 0 gives t=-2.17, **p=0.096 two-sided**
(p=0.048 one-sided, in the predicted direction); sign test on 4/5
negative gives p=0.375 two-sided. **Neither clears a conventional
two-sided threshold at n=5** -- five sites is too few for this sweep
alone to reach the statistical certainty of the original 5-seed,
single-site result, and it isn't being reported as if it did. The value
of this check is corroboration of direction and rough magnitude across
sites the model has never seen, not a fresh significance claim: on that
reading, the result is "mostly yes, not universally" -- exactly the
outcome the check was designed to be able to report either way.

**Bottom line on both checks, stated plainly**: the -0.055-ish AUROC drop
survives prevalence-matching decisively and generalizes across most
(4/5), but not all, held-out sites. The original arm E number is a real
and largely representative measurement of a site-generalization cost,
not a prevalence artifact and not purely an artifact of which single
site happened to land in the original split -- but it is not a universal
per-site guarantee either, and a claim that *every* held-out site would
show this drop would be overstated by this data.

### Follow-up D: is site_4 within a small test fold's noise, and does distance-from-training explain it?

Two more checks on the one exception (site_4, +0.0208), both in
`notebooks/site4_followup.py`, both reusing the leave-one-site-out
results above without any retraining.

**Check A -- a confidence interval on each site's gap that accounts for
test-fold size, not just seed-to-seed variance.** The original 5-seed
Bonferroni analysis (Follow-up B) answers "does this gap reproduce
across differently-trained models on this *same fixed* test fold?" --
yes, decisively. It does not answer a different question: "would this
gap's sign and rough size survive swapping in a different, equally-sized
sample from that site's population?" -- a question that gets harder to
answer the smaller the fold. Closed-form Hanley-McNeil (1982) SE for a
single AUC estimate, using only the AUC value and the fold's fixed
n1 (abnormal)/n0 (normal) counts -- no raw predictions needed, so no
retraining. Total variance per site = this test-fold-size term (fixed,
does not shrink with more seeds, since every seed sees the identical
fold) + seed-to-seed variance/n_seeds (does shrink). Gap CI combines the
site's total variance with B's own (n1=704, n0=575, its own 3-seed
variance) under independence:

| Held-out site | n1 / n0 | gap | SE | 95% CI |
|---|---|---|---|---|
| site_0 (original) | 1248/734 | -0.0529 | 0.0392 | [-0.1298, +0.0239] |
| site_1 | 245/256 | -0.0334 | 0.0604 | [-0.1518, +0.0850] |
| site_2 | 155/249 | -0.0368 | 0.0705 | [-0.1750, +0.1014] |
| site_3 | 186/193 | -0.0339 | 0.0669 | [-0.1650, +0.0971] |
| site_4 | 152/184 | +0.0208 | 0.0759 | [-0.1278, +0.1695] |

**Every single interval spans zero -- including the original site_0
result.** This is not a contradiction of the earlier Bonferroni-
significant finding; it is a different, more conservative question
answered honestly. The original claim (reproduces across models on a
fixed fold) still stands unchanged -- it is about model reproducibility,
not population generalization, and this check does not touch it.
This new check says: with a single site's worth of held-out images (336
to 1982, one fold each, not a random sample repeated many times), none
of the five per-site AUROC estimates is precise enough on its own to
rule out a true population gap of zero for that specific site -- **the
sign flip at site_4 sits comfortably inside noise this test-fold size
can produce, and so, by the same honest standard, does every other
site's negative gap when judged this way.** Reported exactly that
plainly: this check *weakens* confidence in reading any single site's
number literally, symmetrically across all five, not just site_4 --
the right correction is more sites and/or bigger folds per site, not a
larger claim from the same data.

**Check B -- does distance from the training distribution predict the
accuracy loss?** Hypothesis (stated before computing anything): sites
further from the training pool in embedding space should show a larger
accuracy drop, and site_4, being the least-anomalous gap, should be the
*closest*. Mean pairwise cosine distance (pretrained ResNet18 penultimate
features, `dedupe.compute_resnet18_embeddings` -- the identical
extraction dedupe itself uses, not a re-derivation) from each held-out
site's images to the rest of the dataset (the training pool):

| Held-out site | mean cosine distance to pool | gap |
|---|---|---|
| site_3 | 0.1200 | -0.0339 |
| site_0 | 0.1259 | -0.0529 |
| site_4 | 0.1361 | +0.0208 |
| site_1 | 0.1367 | -0.0334 |
| site_2 | 0.1374 | -0.0368 |

**The hypothesis does not hold.** site_4 is not the closest site to the
training pool -- it's tied with site_1 near the *farthest* end, and
site_0 (the largest accuracy loss) is the *second-closest*, the opposite
of what the hypothesis predicts. Pearson r=+0.401 (p=0.503), Spearman
r=+0.100 (p=0.873) between distance and gap -- weak, in the wrong
direction to support the hypothesis even before accounting for n=5
making any correlation here untestable. Reported as a genuine negative
result, not reframed: distance-from-training-in-this-embedding-space
does not explain site_4 (or the sweep's pattern generally). Worth noting
as a limitation of the check itself, not just the hypothesis: all five
sites' distances sit in a narrow 0.117-0.137 band -- unsurprising, since
"site" here is a resolution-cluster proxy within one dataset's fundus
photography, not genuinely different imaging domains, so this specific
metric may simply lack the dynamic range to discriminate a real effect
even if one exists. **site_4 remains an unexplained exception** --
reported as exactly that, per the instruction that motivated this whole
check: an unexplained exception, honestly reported, is a fine outcome.

## Why curation costs AUROC: three hypotheses tested, the flattering one lost

The original writeup offered one hypothesis for C's -0.0146 AUROC vs B:
quality curation removed images that were hard-to-grade but not
uninformative to a CNN. Plausible, and flattering to report without
checking further (it makes curation sound sophisticated). Tested three
alternatives in order of confidence instead of taking it on faith.

**(a) Quality correlates with disease -- tested and refuted, in the
opposite direction from the hypothesis.** If cataract/media opacity both
cause haze and constitute pathology, rejects should skew abnormal.
Checked directly: of the 43 rejects, 27 are normal (62.8%) and 16 abnormal
(37.2%), against a dataset baseline of 45.0%/55.0% -- **rejects skew
*normal*, not abnormal** (chi-square vs baseline: p=0.019, real
difference, wrong direction for the hypothesis). Population-scale check,
much stronger than 43 images can give: mean gradability score by label
across the full 6392-image dataset is 0.8482 for normal, 0.8553 for
abnormal -- abnormal images score very slightly *higher* quality on
average (t-test p=0.0039, Mann-Whitney p=0.041; the effect is small,
0.007, but real at this n). **This hypothesis does not hold.** Removing
the 43 rejects shifts overall class balance from 44.96% normal to 44.85%
-- negligible, and in a direction that doesn't explain a meaningful AUROC
drop. Speculative, not established: one possible reason rejects skew
normal here is that images collected as part of a confirmed-pathology
workup may have received more careful capture than routine/screening
normals, but this dataset doesn't have the metadata to test that, and no
claim is made about fundus imaging in general from a finding this
dataset-specific.

**(b) Composition, not size -- tested and confirmed, much larger than
expected.** Matched training-set *size* (4435 for both B and C) does not
mean matched training-set *identity*. Checked directly: B's and C's
actual training sets share only **3071 of 4435 images (69.2%)** -- 1364
images (30.8%) are simply different, not because they were quality-
rejected (only 43 were), but because `patient_group_split` is recomputed
fresh via `StratifiedGroupKFold` on each arm's own curated pool, and that
algorithm reassigns a large fraction of fold membership even from a small
change to its input.

**(c) Magnitude sanity check -- resolved by (b), and the "1%" framing was
misleading.** Is a 0.0146 AUROC shift plausible from removing under 1% of
the data? Wrong question: **the real perturbation is 30.8% of the
training set, not 0.97%.** Confirmed with a control: removed 43 *random*
images, uncorrelated with quality or content, from the same manifest, and
recomputed the split the same way -- **69.4% overlap with B's original
training set, statistically indistinguishable from the real 69.2%.**
Any 43-image removal from this pool causes almost exactly this much
reshuffling; the quality-relatedness of the actual 43 removed images is
incidental to this specific mechanism, not the cause of it.

**Revised leading explanation, replacing the original hypothesis**: the
C-vs-B AUROC difference is best explained by `StratifiedGroupKFold`'s
sensitivity to small perturbations in its input pool, not by curation
removing informative content. Matching training-set *size* across arms
(which this project does carefully) is not sufficient to isolate a
curation effect when the split itself is recomputed per arm -- roughly
a third of the training set differs between B and C regardless of *why*
the pool changed. This is a real methodological finding about this
project's own arm-comparison design, not just about ODIR-5K: a cleaner
design for isolating curation's effect specifically would compute the
split once on the full raw pool and then remove quality-rejected images
from whichever fold they land in, rather than re-splitting the curated
pool from scratch -- flagged here as a design lesson for future work, not
implemented now (this is a report-only investigation, not a pipeline
change). The original "informative-but-hard-to-grade" hypothesis isn't
disproven outright -- it could still be a minor contributing factor
layered on top of the reshuffling effect -- but it is no longer the
leading account, and it should not have been reported as the sole
explanation without this check.

## Variance instability of the naive split -- investigated, did not replicate either

Arm A's AUROC std in the second (4-arm) run is 0.0129 vs arm B's 0.0028
-- a 4.6x ratio, and a real, mechanistically sensible second argument for
grouped splitting (the naive split's score depends on which patients
happened to straddle the fold boundary, so beyond being optimistic on
average it could be less *stable* run to run). Checked against the first
A/B-only run before promoting it, the same way the mean-difference finding
was checked. **It does not hold up the same way across both runs:**

| Run | Metric | std A | std B | Ratio (A/B) | Levene p |
|---|---|---|---|---|---|
| First (cap=4473) | AUROC | 0.0069 | 0.0083 | 0.83x | 0.729 |
| First (cap=4473) | AUPRC | 0.0052 | 0.0060 | 0.86x | 0.711 |
| First (cap=4473) | Sens@95 | 0.0262 | 0.0102 | 2.58x | 0.140 |
| Second (cap=4435) | AUROC | 0.0129 | 0.0028 | 4.58x | 0.081 |
| Second (cap=4435) | AUPRC | 0.0117 | 0.0032 | 3.59x | 0.142 |
| Second (cap=4435) | Sens@95 | 0.0252 | 0.0176 | 1.43x | 0.540 |
| Pooled (n=10/arm, informal) | AUROC | 0.0107 | 0.0059 | 1.82x | 0.084 |
| Pooled (n=10/arm, informal) | AUPRC | 0.0090 | 0.0046 | 1.97x | 0.195 |
| Pooled (n=10/arm, informal) | Sens@95 | 0.0249 | 0.0163 | 1.53x | 0.192 |

In the first run, A was actually *less* variable than B on AUROC/AUPRC
(ratio <1) -- the opposite of the hypothesized direction -- and only
Sens@95%Spec showed A more variable (2.58x), the metric that shows the
*weakest* version of the pattern in the second run (1.43x). No individual
run's Levene test reaches p<0.05 for any metric; the informally pooled
10-observations-per-arm estimate (ratio ~1.8-2x for AUROC/AUPRC) is
directionally suggestive but still not significant (p=0.08-0.20) at this
sample size, and pooling itself is a caveat-laden move since the two runs
used slightly different training-set caps (4473 vs 4435), not identical
conditions.

**Honest verdict, stated as plainly as the finding it was checked
against**: this does *not* clear the bar this project has held every
other claim to. It is directionally present in the second run and in one
metric of the first, but it does not replicate cleanly, exactly like the
mean-difference finding it was hoped to reinforce. Reporting it as "a
result that replicated when the headline didn't" would not be true to
what checking it found -- so it isn't reported that way. It is reported as
an investigated, suggestive, unconfirmed lead: worth more data (more
seeds) to resolve, not worth a confident second argument for grouped
splitting in the README's headline alongside the (also unconfirmed)
mean-difference result.

## Arms C and D: full 4-arm run -- the prediction partly held, and A-vs-B didn't replicate

Full dataset, 5 seeds (42-46), epochs=15, GPU, train size matched across
all four arms (cap=4435, computed fresh from A/B/C/D's actual curated
pools -- natural sizes were A=4473, B=4474, C=4443, D=4435). Total
wall-clock ~3h57m (A: 57.7min, B: 84.5min -- this arm overlapped with an
unrelated ~50min CPU-contention stall from a concurrent analysis task
requested mid-run, see below, C: 56.7min, D: 37.9min).

Mean +/- std across 5 seeds (ddof=1):

| Arm | AUROC | AUPRC | Sens@95%Spec |
|---|---|---|---|
| A (image_random) | 0.8014 +/- 0.0129 | 0.8512 +/- 0.0117 | 0.4446 +/- 0.0252 |
| B (patient_group) | 0.7936 +/- 0.0028 | 0.8450 +/- 0.0032 | 0.4207 +/- 0.0176 |
| C (quality) | 0.7790 +/- 0.0044 | 0.8382 +/- 0.0058 | 0.4248 +/- 0.0284 |
| D (quality+dedupe) | 0.7820 +/- 0.0117 | 0.8404 +/- 0.0086 | 0.4495 +/- 0.0183 |

### The A-vs-B gap did not replicate on this run -- reported as prominently as the original finding

Same nominal seeds (42-46), same split strategies, matched training size
within half a percent of the original A/B-only run (4435 vs 4473) -- and
this time: mean AUROC diff (A-B) = **+0.0079** (was +0.0173), t=1.23,
**p=0.29** (was p=0.037), sign 3/5 favouring A (was 5/5). Neither
uncorrected significance nor sign-consistency survived. This is not a
contradiction or an error to explain away -- it is exactly what n=5 being
underpowered *means*, demonstrated empirically by an actual second draw
rather than argued from a power calculation. The original write-up
already said the Bonferroni-corrected CIs included zero and that this
wasn't a settled result at n=5; this run is that caveat made concrete.

One technical note, so the replication is understood precisely rather
than treated as an unexplained mystery: this is not a literal
same-seed-same-result puzzle. `_cap_train_size` samples down to the cap
using a Generator seeded on `cfg["seed"]` (always 42), but the cap itself
changed (4473 -> 4435, because *this* run had to match across all four
arms, not just two) -- `rng.choice(n, size=cap, ...)` draws a different
specific subset when `size` changes, even from an identical seed. So the
two A runs trained on ~99.15% overlapping but not identical image sets.
That's a real, understood difference, not nothing -- but it's a ~38-image
(0.85%) perturbation, and it was enough to flip the result from
"significant, 5/5 sign-consistent" to "not significant, 3/5." If a
perturbation this small can do that, n=5 was already living right at the
edge of noise before this run confirmed it.

### C vs B and D vs B, same rigor as the A-vs-B writeup

No a priori direction was predicted for these two (unlike A-vs-B, where
the leakage hypothesis justified a one-sided test) -- curation could
plausibly help, hurt, or do nothing, so these are two-sided by default.
Family of 6 tests (3 metrics x 2 comparisons): Bonferroni alpha =
0.05/6 = 0.00833, t_bonf(df=4) = 4.851.

| Comparison | Metric | Mean diff | t | p | 95% CI | Bonferroni CI | Sign |
|---|---|---|---|---|---|---|---|
| C - B | AUROC | -0.0146 | -4.93 | **0.0078** | [-0.0228,-0.0064] | [-0.0290,-0.0002] | 0/5 (all negative) |
| C - B | AUPRC | -0.0068 | -2.11 | 0.1025 | [-0.0157,+0.0021] | [-0.0223,+0.0088] | 0/5 (all negative) |
| C - B | Sens@95 | +0.0041 | 0.36 | 0.7405 | [-0.0278,+0.0360] | [-0.0517,+0.0599] | 3/5 |
| D - B | AUROC | -0.0116 | -1.84 | 0.1403 | [-0.0290,+0.0059] | [-0.0421,+0.0190] | 1/5 |
| D - B | AUPRC | -0.0046 | -0.88 | 0.4265 | [-0.0191,+0.0099] | [-0.0299,+0.0207] | 2/5 |
| D - B | Sens@95 | +0.0288 | 1.80 | 0.1456 | [-0.0155,+0.0730] | [-0.0486,+0.1061] | 4/5 |

**D vs B: the prediction held.** No metric significant, uncorrected or
corrected; signs are mixed on AUROC/AUPRC. Consistent with "43+22 removed
images can't plausibly move a metric with this much seed noise."

**C vs B: the prediction did not fully hold, and this is the surprising
result requiring explanation, stated as such rather than smoothed over.**
AUROC is lower for C than B, by a small but real amount (-0.0146,
~1.5 points), *consistently* (5/5 seeds, every one negative -- the
tightest sign-consistency in this entire project), and the difference
**survives Bonferroni correction at the 6-test family level** (p=0.0078
< 0.00833) -- though only just: the corrected CI's upper bound is
-0.0002, a hair below zero. AUPRC points the same direction but does not
reach significance either way. Sens@95%Spec shows no effect.

Why this is plausible rather than a red flag: matching training-set
*size* across arms (which this project does carefully) does not match
training-set *composition*. Quality curation removed the 43 lowest-scoring
images specifically -- the blurriest, darkest, most uneven-illumination
examples in the dataset (see the quality.py section below). It's a
real, testable hypothesis that some of those images, while genuinely
harder to grade by a human-legible standard, were not *uninformative* to
a CNN -- removing them changes what the model sees during training in a
way that "curation should only help or do nothing" doesn't predict, and
arm B and arm C both happen to have unusually low between-seed variance
(0.0028 and 0.0044 respectively, versus 0.0129/0.0117 for A/D), which is
exactly the condition under which a small, real, consistent effect
becomes statistically detectable rather than lost in noise. This is a
hypothesis, not a proven mechanism -- not tested further here, but a
concrete, falsifiable one for future work, not a hand-wave.

**Net honest verdict, stated in advance and checked against reality**: the
predicted null (quality/dedupe curation is too small to move AUROC) held
for D and did not fully hold for C, where a small, sign-consistent,
marginally-significant *decrease* was found instead of the predicted null.
Curation earning its place by catching a real integrity problem (the 8
duplicate pairs) rather than by improving a metric turned out to be
literally true for D, and turned out to be worth stating even more plainly
for C, which moved a metric -- just not in the direction curation is
usually assumed to move it.

## dedupe.py: phash verified properly this time, embedding threshold was also wrong

Implemented per CLAUDE.md step 6: phash candidates + pretrained-ResNet18
embedding candidates, both against the real 6392-image dataset.

**phash, done right:** the earlier ad hoc investigation (see the label/
dedupe finding above) only pixel-diff-verified the strictest hamming==0
bucket (14 candidates, found 2 genuine pairs). The real module verifies
*every* phash candidate at the configured hamming<=6 (13733 of them) by
actual pixel difference, not just the tightest bucket -- and found **8
genuine pairs**, not 2. Two additional real duplicate pairs (398 vs 668,
3297 vs 4542) were sitting at hamming 1-6, missed by only checking
hamming==0. Lesson: verify every candidate, not just the ones under the
strictest sub-threshold.

**embedding_cosine_min: 0.98 (the config default) was also wrong, caught
before trusting the output.** First run at 0.98 produced 165 candidate
pairs, and the audit logged nearly all of them as "cross-patient integrity
issues" -- a volume as suspicious as the earlier FOV-clipping numbers, so
checked before believing it. Pixel-difference verification (the tool that
worked for phash) doesn't apply here: embeddings are explicitly meant to
catch pixel-*different* content (same eye, different lighting), so a
tight pixel-diff gate would reject genuine embedding finds. Verified by
eye instead: `28_right.jpg` vs `32_right.jpg` (sim 0.98, pixel-diff 12.4)
-- plausibly genuine, nearly identical vessel branching, different
color/exposure. `180_right.jpg` vs `394_right.jpg` (sim 0.98, pixel-diff
8.9) -- clearly two different patients, different vessel topology
entirely. At 0.98 the threshold also caught 4 genuine fellow-eye pairs
(patients 1032, 2110, 2217, 2219) as false "duplicates" -- exactly the
failure mode flagged as a risk in advance. Checked cosine_min against
fellow-eye false positives and candidate count directly: 0.98 -> 165
pairs/4 fellow-eye FPs, 0.99 -> 10 pairs/0 FPs, 0.995 -> 5, 0.998 -> 4.
**Raised to 0.99**: eliminates every fellow-eye false positive, and all 10
remaining candidates have low pixel-diff (0.00-7.74) with the 3 not
already found by phash (`31_left` vs `105_left`, `1109_right` vs
`1166_right`, `4330_left` vs `4552_left`) visually confirmed -- the first
two are unmistakably the same eye (near-identical vessel branching), not
a plausible call.

**Empirical justification for 0.99, not just "raised until false
positives went away"**: computed cosine similarity for a 400-patient
random sample of genuine fellow-eye pairs (CPU, to avoid any GPU
contention with the arm C/D training run in progress) and for all 8
confirmed duplicate pairs (11 pair-instances, 3 duplicated on both eyes).

| | n | mean | min | max | 99.5th pct |
|---|---|---|---|---|---|
| Fellow-eye (sampled) | 400 patients | 0.933 | 0.770 | 0.975 | 0.973 |
| Confirmed duplicates | 11 pairs | 0.993 | 0.962 | 1.000 | -- |

This is *why* 0.98 caught fellow eyes and 0.99 mostly doesn't: a patient's
two eyes are genuinely more similar to a generic ImageNet-pretrained
ResNet18 than two random retinas are (bilateral anatomical/pigmentation
similarity, not just chance) -- the fellow-eye distribution's own 99.5th
percentile (0.973) sits close under 0.98, so a threshold anywhere near
0.98 was always going to clip into the top of that distribution. 0.99
clears it with room.

**But the separation is not clean, and that matters more than the
summary stats above suggest**: one confirmed duplicate, 3297 vs 4542
(pixel-diff verified via phash), has cosine similarity **0.962** --
*below* the fellow-eye sample's own maximum (0.975) and 99.5th percentile
(0.973). No embedding threshold, at any value, would have caught this
pair without also catching genuine fellow eyes -- it was only found
because phash's independent, pixel-structure-based signal doesn't care
that this particular duplicate's two copies apparently differ enough in
surface appearance (lighting/color grading) to fool a semantic embedding.
**This is the concrete, not-just-theoretical proof that both methods are
structurally necessary**: embeddings alone would have found only 7 of the
8 patient-pairs, permanently blind to 3297/4542, confirmed directly
above. **Correction, caught in a later review pass: this section
originally also claimed "phash alone would have found all 8" -- wrong,
and not checked at patient-pair granularity before being written down,
which is exactly the kind of unverified number this project exists to
catch.** Phash's own image-pair count genuinely is 8, but those 8
image-pairs cover only 5 of the 8 patient-pairs in full -- phash never
verified any image-pair for 31/105, 1109/1166, or 4330/4552 (each a
single-eye duplicate, so missing their one cluster means missing the
whole patient-pair), confirmed by re-checking hamming distance and
pixel difference directly against the real images rather than trusting
the original count. Both methods have real, independent blind spots on
this data, and neither's blind spot is visible from inside the other --
see WALKTHROUGH.md §7 for the connection to the phash false-positive
finding, which is the same underlying pattern from the opposite side.

**Final result**, phash-verified (8 image-pairs) + embedding at the
corrected 0.99 threshold (10 image-pairs, 7 overlapping with phash):
**11 distinct duplicate clusters, 22 images, across 8 unique
patient-pairs** (3 pairs duplicated on both eyes, 5 on one eye). None
are same-patient/fellow-eye; all are cross-patient -- the same real
capture filed under two different declared patient IDs.

**Money metric**: of these 11 unique verified pairs (8 phash + 10
embedding, 7 found by both), **3 straddle the `image_random` split's
fold boundaries; 2 straddle `patient_group`'s.** Materially the same
order of magnitude for both -- confirms directly what the design
argument already predicted: patient-grouped splitting has no mechanism
to catch a duplicate filed under two *different* patient IDs, since it
only keeps a single declared ID's images together. This class of
leakage is dedupe's job specifically, not splitting's.

**Correction, caught in a later review pass: this section originally
reported 4/3 straddling image_random/patient_group, from "18 verified
pairs (8+10, double-counting the 6 found by both methods)."** The bug
was in `run_dedupe`, not in this document alone: it fed the raw
concatenation of phash-tagged and embedding-tagged pairs straight into
the cross-split straddle count, so a pair found by *both* methods was
counted as two straddling events instead of one, whenever it happened
to straddle a boundary -- the same shape of error as the cluster-overlap
count above (originally "6," corrected to "7"), except this instance had
a real downstream consequence (an inflated persisted artifact) rather
than being a documentation-only slip. Fixed by deduplicating
method-tagged pairs into unique real-world pairs
(`dedupe.deduplicate_across_methods`) before any cross-method count, with
a regression test (`tests/test_dedupe.py`) constructing a pair found by
both methods and asserting the straddle count treats it once.
`artifacts/duplicates_cross_split.json` regenerated: **3** for
`image_random`, **2** for `patient_group` -- confirmed by re-running
`retinaprep dedupe` against the real dataset.

**Prediction, stated before running arms C and D**: quality curation
removes 43 images (0.7%) and deduplication removes at most 22 (0.3%, and
only from whichever side of each pair curation decides to drop). Neither
change is large enough to plausibly move an AUROC measured with a paired
seed-to-seed noise floor of ~0.05 (the A/B result above). **Arms C and D
are expected to be statistically indistinguishable from arm B.** If they
come out otherwise, that is the surprising result requiring explanation,
not the expected one -- these stages earn their place in the pipeline by
being the right thing to do and by catching the genuine duplicate-ID
integrity problem above, not by moving a metric on a dataset that turned
out to already be this clean.

## quality.py sanity checks: false rejects, uniform haze, raw-vs-preprocessed

Three checks requested after the 0.7% reject rate landed, to make sure that
number meant what it looked like it meant rather than being trusted at
face value.

**1. Inspected all 43 rejects by eye** (contact sheet, sorted by score).
Clear majority (roughly the bottom half, scores <0.43) are genuinely poor
-- very dark, hazy, indistinct, no real disagreement possible. A handful
near the 0.45-0.50 boundary (e.g. `716_left.jpg`, the oval-cropped image
from the FOV investigation above, at 0.493) show enough visible vessel/disc
structure that a human grader might call them borderline-gradable rather
than clearly ungradable. That is expected behaviour at any hard cutoff on
a continuous score, not evidence of a scoring bug -- no clearly-good image
(comparable to the 0.99 examples) was found sitting in the reject pile.
**No false rejects found; some genuinely ambiguous near-boundary cases,
correctly described as such rather than as clean failures.**

**2. Uniform haze (the cataract case) -- confirmed as a real, unfixed
limitation.** Checked the specific image flagged during the FOV
investigation as visually hazy (`1281_left.jpg`): score **0.93**, 81st
percentile. Searched further using mean HSV saturation as a haze proxy
among score>=0.85 images (haze desaturates colour) and found worse cases
by eye: `1264_right.jpg` (score **0.943**) and `2123_left.jpg` (score
**0.979**, near the top of the entire dataset) are both essentially
featureless uniformly hazy/foggy images -- no visible vessels, no visible
disc, consistent with dense cataract or severe media opacity. Both should
be clinically ungradable and both score near-perfect.

Why the module misses this: `illumination_uniformity` measures *uneven*
illumination (one side dark, one side bright) -- uniform haze is, by
definition, uniform, so it reads as good. Blur metrics stay above
threshold because moderate haze doesn't eliminate all high-frequency
content (compression artifacts, faint specular reflections still register
on Laplacian/Tenengrad), even when the actual retinal structure a
clinician needs is gone. There is no metric here that measures *global
contrast* or "is there actually vessel structure to see," which is what
would be needed to catch this. **Stated plainly as a known limitation, not
fixed in this pass**: this quality module reliably catches blur, under/
over-exposure, and uneven illumination; it does not reliably catch
uniform haze. A real deployment would need an added contrast/entropy-style
check (or a learned gradability classifier) specifically for this failure
mode -- classical per-quadrant/per-pixel statistics of the kind used here
structurally cannot distinguish "uniformly hazy" from "uniformly clear."

**3. Raw Training Images/ comparison, to calibrate what 0.7% means.**
`preprocessed_images/` (6392 files) is a subset of the raw pool (7000
Training + 1000 Testing = 8000) -- roughly 20% of raw captures never made
it into the curated set at all, before this project's own quality.py ever
ran. Scored a random 400-image sample of the raw `Training Images/`
folder with the identical code path (`compute_metrics` + `gradability_score`,
same config thresholds): **17/400 = 4.2% reject rate**, against 0.7% on
the already-curated `preprocessed_images/`. A 6x relative difference on
the same thresholds, same code, different population -- the 0.7% number
reflects that this dataset was already curated upstream, not that the
module is lax. Reported at the sample size actually run (400 of 7000);
the direction and rough magnitude are the useful signal here, not a
precise population estimate.

## quality.py: preprocessed_images/ sizing, and the FOV-clipping threshold that wasn't real

Checked directly before implementing: `preprocessed_images/` is uniformly
512x512 (200-image random sample, zero exceptions), so the classic
resolution-dependence trap for blur metrics is moot for what this pipeline
actually scores. The raw `Training Images/` mixes 12+ distinct resolutions
(894px-3888px) in the first 50 files alone, so `variance_of_laplacian` and
`tenengrad` still resize to a fixed size before scoring unconditionally --
right for that folder and for any future dataset this module gets pointed
at, even though it's a no-op here.

**First implementation hard-rejected 51.1% of the dataset.** Diagnosed
before reporting that number: it was not the weighted score (blur/exposure/
illumination) -- that alone rejected only 0.5% of FOV-valid images, which
looked sane. It was a hard "FOV clipped" gate checking whether the *fitted
enclosing circle's* geometric extent exceeded the frame. `fov_radius_frac`
has median 0.998 across all 6392 real images -- the FOV is pre-cropped to
fill the frame almost exactly -- so ordinary off-centre fitting noise on a
real (non-perfectly-circular) photograph triggers that check on the
majority of completely normal images. Not a threshold that needed
retuning; a wrong check.

Fixed once (contour-touches-border instead of fitted-circle-exceeds-frame)
-- reject rate dropped to 38.5%, still absurd. Investigated the remaining
gate visually rather than re-tuning blind: pulled real examples across the
circularity range (contour_area / (pi * enclosing_radius^2), which should
be ~1.0 for a genuine full circle). Findings, all confirmed by eye:
- circ ~0.96: near-perfect circle, trivially fine.
- circ ~0.90: a genuinely bad image (hazy/media-opacity artifacts) -- but
  that's a quality problem the score should catch on its own merits, not
  evidence of frame clipping.
- circ ~0.85: real truncation -- a visible straight-edge cutoff.
- circ ~0.80 and ~0.70: **complete, clean, fully gradable images that are
  oval-cropped rather than circular** -- disc, macula and vessels all
  clearly visible, nothing missing. Not clipped at all; just not a circle.

Circularity conflates three different things on this dataset (real
truncation, complete oval crops, and hazy images) and cannot cleanly
separate them with a single threshold -- confirmed by direct visual
inspection at several threshold values, not assumed. Forcing it to gate
rejection punishes the common, legitimate oval-crop case as if it were
damage. **Fix:** `fov_clipped` is still computed and stored in
`quality.parquet` (circularity < 0.85) for visibility, but no longer gates
`fov_valid` or the score. Only `fov_radius_frac_min` (a real, uncontroversial
"is the field too small" check) gates FOV validity now. The transparent
weighted score is what actually determines gradability -- which is also
more useful in practice: it says blur, exposure, or illumination was the
problem, rather than an opaque "FOV rejected" that, as built, was usually
wrong about why.

**Final reject rate: 43/6392 = 0.7%**, well under the ~10% concern
threshold -- reported as measured, no further tuning attempted once the
number looked sane and matched the (unforced) weighted-score-only rate
from the diagnosis step. Contact sheet of the 20 lowest- and 20
highest-scoring images (`artifacts/quality_contact_sheet.png`) confirms
the score direction by eye: the lowest 20 show visible blur, haze, or
uneven illumination; the highest 20 (all ~0.99) are crisp, well-exposed,
with clearly visible vessels and disc. There are no ground-truth quality
labels for this dataset, so this eyeball check against the extremes is the
honest substitute for validation, per the same principle applied to the
dedupe threshold investigation above.

## Why is the measured effect (0.017 AUROC) smaller than 42.3% patient overlap suggests?

**[Correction, added later: the "42.3%" in this section's title and body was
computed under the `normal_column` label strategy, before the switch to
`keywords` (see "Label strategy default switched" above) -- it was never
re-derived after the switch and went stale. Re-run from a clean
`artifacts/` directory with the current default config: **42.0%
(1412/3358)**, verified byte-identical across two independent runs. This
does not change the substance of the investigation below (the dilution
mechanism, and the 77.8%/50.5% concordance numbers, are unaffected -- they
depend on the label column, not the split), only the specific overlap
percentage quoted alongside it. Left as originally written below, with this
note, rather than silently edited, so the correction itself is part of the
record -- see README.md's headline section for where the corrected number
now lives.]**

Report-only investigation, no pipeline changes, no training re-run. Three
candidate explanations tested against evidence, plus the duplicate-pairs
check originally planned for dedupe.py.

### 1. Bilateral correlation (concordance rate) -- best-supported explanation

Computed directly from the real data (keywords label strategy, reusing the
actual adapter's `_derive_eye_column`/`_label_from_keyword_string`): among
3034 multi-eye patients, fellow eyes share the same binary label for 2360
of them -- **77.8% concordant, 22.2% discordant**.

The right comparison isn't 77.8% vs. 100% -- it's 77.8% vs. what chance
alone would give at this class balance (44.96% normal / 55.04% abnormal):
P(both normal) + P(both abnormal) = 0.4496^2 + 0.5504^2 = **50.5% by chance
alone**. So there is a real, substantial positive bilateral correlation
(77.8% vs. a 50.5% chance floor) -- but it is far short of the ~100% that
would make leaking a patient's identity equivalent to leaking their label
outright. For 22.2% of multi-eye patients, knowing one eye's label is
actively uninformative or misleading about the fellow eye's.

This directly explains the gap between "42.3% of patients leak across
folds" and "only ~0.017 AUROC of measured effect": leaking a patient's
identity does not mean leaking their label. It means leaking a
noisy, ~78%-reliable hint about it, and only for the subset of leaked
patients whose fellow-eye image actually ends up as an exploitable
train/test pair, and only insofar as an 8-epoch ResNet18 actually learns
and uses that hint (see #3). Every stage dilutes the raw overlap
percentage; this stage alone caps the theoretical maximum benefit of
leakage at roughly the concordance rate, not at 100%.

### 2. Task coarseness -- plausible aggravating factor, not tested

Class balance under the actual label strategy (keywords) is 44.96% normal
/ 55.04% abnormal -- reasonably balanced, not the dominant issue on its
own. But binary normal/abnormal collapses 8 original ODIR categories
(N/D/G/C/A/H/M/O) into one "abnormal" bucket. Two different diseases in
two different eyes both count as "abnormal" and therefore "concordant"
under this task, even though they are unrelated conditions and a
per-disease task would score them as discordant. Conversely, a specific
disease (e.g. diabetic retinopathy, driven by systemic disease) is
plausibly *more* bilaterally consistent than "any of 7 possible
abnormalities," so a per-disease or multi-class task would likely show
*higher* concordance and therefore a *larger* leakage effect than measured
here. This is a real, plausible reason the binary task under-states the
true leakage risk -- but it is an assessment, not a measurement (not run,
per instruction), so it stays a hypothesis, not a finding.

### 3. Model capacity -- evidence points away from this being the bottleneck

Extracted the full per-epoch val AUROC trajectory for all 10 runs (5
seeds x 2 arms) from the training log. Pattern across nearly every run:
val AUROC rises for several epochs, peaks, then **declines** before
patience (3) fires -- e.g. A seed42 peaks at epoch 5 (0.788) then falls to
0.752 by epoch 8; A seed44 peaks at epoch 7 (0.784) then drops sharply to
0.727 by epoch 10. This is the opposite of "still climbing when cut off":
in most runs the best epoch is not the last one, and the tail after the
peak is a real decline, not just noise sitting near the peak. That pattern
argues *against* model capacity/undertraining being the main bottleneck --
if the model were too weak to have exploited available leakage signal yet,
truncated improvement (monotonic rise, cut short by patience) is what we'd
expect to see, not rise-then-fall. It doesn't rule out a different
architecture extracting more signal, but "ResNet18 just needed more
epochs" is not what these curves show.

Caveat shared with the epoch-variance finding already in this document:
the curves are noisy and non-monotonic throughout (consistent with
patience triggering on fluctuation), which makes any single verdict here
less clean than the concordance measurement above. Treated as suggestive,
not conclusive.

### Duplicate pairs (the check originally planned for dedupe.py)

phash (imagehash, 64-bit) computed for all 6392 images. At the **configured**
threshold (`dedupe.phash_hamming_max: 6`): 13733 near-duplicate pairs,
almost entirely cross-patient (13730 of 13733). That number is not what it
looks like -- visually inspected several flagged pairs (e.g. patient 0 vs
549, patient 156's own left vs right eye) and they are clearly different
photographs that merely share fundus photography's generic macro-structure
(dark background, circular FOV, similar framing). **hamming<=6 is far too
loose for this domain and produces a large false-positive rate; it should
not be trusted as configured.** This also directly answers the fellow-eye
concern: 3 same-patient (fellow-eye) pairs were flagged at this threshold,
and the one inspected (patient 156) is visibly not a duplicate -- the
threshold is too loose in exactly the way that would incorrectly flag
fellow eyes, confirming the config's own warning needs heeding before
dedupe.py is built for real.

Tightening to the strictest possible bucket, hamming==0 (bit-identical
64-bit hash): 14 pairs, still 0 same-patient. Verified each with actual
pixel difference (mean absolute difference per pixel, 0-255 scale), since
even a perfect hash match is a probabilistic signal, not proof:

- **2 pairs are genuine duplicates**, confirmed on both eyes: patient 352
  vs 973 (mean abs diff 0.00 and 0.01 on right/left) and patient 2487 vs
  3185 (0.01 and 0.00). Different MD5 (different JPEG encoding) but
  pixel-identical content on decode -- the same underlying photograph,
  filed under two different patient IDs. **This is the genuine dataset
  integrity problem flagged loudly, as instructed**: two pairs of distinct
  "patients" in this dataset are, going by their images, almost certainly
  the same real capture duplicated across IDs.
- The other 12 (11 remaining pairs, one already double-counted above) are
  hash collisions with real pixel differences of 15-60/255 -- false
  positives, confirming even hamming==0 needs a secondary check (pixel
  diff or an embedding, as CLAUDE.md's dual-method design already
  anticipates) rather than being trusted alone.

**[Superseded -- see "dedupe.py: phash verified properly this time" above,
the entry immediately below the log at the top of this file.]** This ad
hoc pass only pixel-verified phash's single tightest bucket (hamming==0,
14 candidates). The real dedupe.py module verifies every phash candidate
at the configured hamming<=6 (13,733 of them) plus every embedding
candidate, and found **8** genuine pairs, not 2 -- this "2" figure was an
artifact of checking only the strictest sub-threshold, not the true count.
The two pairs found here are still correct as far as they go; they are a
subset of the real 8, not wrong, just incomplete.

Checked where the two genuine duplicate pairs land: **both pairs sit in
the `train` fold under both `image_random` and `patient_group`** -- so
this specific duplication happens not to be inflating the current A/B
measurement, but only by chance. Patient-grouped splitting does **not**
protect against this class of leakage: 352 and 973 are different
declared patient IDs, so `patient_group` has no reason to keep them
together -- it would be entirely possible for one copy to land in train
and the other in test under a different seed, and nothing in the current
split logic would catch it. Only deduplication closes this gap; it is a
distinct failure mode from the image-vs-patient-level splitting question
the rest of this project measures. (The full 8-pair money metric --
4/18 straddling image_random, 3/18 straddling patient_group -- is in the
superseding entry above.)

### Recommendation

**Bilateral discordance (22.2%) is the best-supported explanation**, and
matches the hypothesis bet on in advance: it is directly measured (not
assessed or inferred), the chance-baseline comparison (77.8% vs. 50.5%)
makes the dilution mechanism concrete and quantifiable, and it requires no
additional untested assumptions. Task coarseness is a plausible
compounding factor pointing the same direction (binary task likely
*understates* true leakage-sensitivity) but is untested by design. Model
capacity is the least supported of the three -- the training curves show
peak-then-decline, not truncated-still-improving, arguing against
"ResNet18 was too weak to use the leakage" as the story.

Unplanned but material finding: 2 genuine cross-patient duplicate pairs
exist in the raw data (dataset integrity issue, not a splitting-code
issue), and the configured phash threshold is demonstrably too loose for
this image domain -- both are inputs the eventual dedupe.py work needs,
not just this investigation. **(Updated once the real dedupe.py module
ran: the true count is 8 pairs, not 2 -- see the entry above this one.)**

## Full-scale A/B run: 5 seeds, full dataset, GPU (cu126) -- real result

Superseded the exploratory CPU run below. Full dataset (subsample_n=null),
epochs=15, early stopping (patience=3) on val AUROC, batch_size=32 (config
default), amp=false, seeds 42-46. GPU timing check first: batch_size=16
used 0.67GB VRAM/43.5s/epoch, batch_size=32 used 1.13GB/39.0s/epoch (faster
AND still far under the 4GB budget) -- used 32. Worst-case estimate before
running was ~1.85h for all 10 runs; actual wall time was ~64 min, since
early stopping triggered at epoch 6-10 in every run, never reaching 15.

Train-set size confirmed matched before this run: A=4473 images/3003
patients, B=4474/2350, both ~45.0%/55.0% class balance. The 1-image gap is
auto-corrected by match_arm_sizes (log: "Capped train set from 4474 to
4473"). The patient-count difference (3003 vs 2350) is expected, not a
defect -- image-level vs. patient-level grouping need different patient
counts to reach the same image count. So none of the earlier 0.098-AUROC
(n=800, 2 seeds) gap was dataset-difference; that question is now moot
anyway since this full-scale run supersedes it.

Mean +/- std across 5 seeds (ddof=1):

| metric | A (image_random) | B (patient_group) | paired diff (A-B) | paired t-test |
|---|---|---|---|---|
| AUROC | 0.8098 +/- 0.0069 | 0.7926 +/- 0.0083 | 0.0173 +/- 0.0125 | t=3.09, p=0.037 |
| AUPRC | 0.8567 +/- 0.0052 | 0.8450 +/- 0.0060 | 0.0117 +/- 0.0089 | t=2.96, p=0.042 |
| Sens@95%Spec | 0.4554 +/- 0.0262 | 0.4378 +/- 0.0102 | 0.0176 +/- 0.0349 | t=1.13, p=0.323 |

Per-seed values (the mean/std above compress this -- this is the actual spread):

| seed | A AUROC | B AUROC | A AUPRC | B AUPRC | A Sens@95 | B Sens@95 | A stop epoch | B stop epoch |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.8151 | 0.7895 | 0.8636 | 0.8418 | 0.4943 | 0.4233 | 8 | 10 |
| 43 | 0.8155 | 0.7815 | 0.8564 | 0.8386 | 0.4361 | 0.4517 | 7 | 8 |
| 44 | 0.8105 | 0.7949 | 0.8560 | 0.8424 | 0.4290 | 0.4375 | 10 | 7 |
| 45 | 0.7984 | 0.7925 | 0.8492 | 0.8487 | 0.4503 | 0.4403 | 8 | 10 |
| 46 | 0.8097 | 0.8044 | 0.8585 | 0.8536 | 0.4673 | 0.4361 | 6 | 8 |

**Early-stop epoch, out of a 15-epoch budget (patience=3): A ranges 6-10
(std 1.48), B ranges 7-10 (std 1.34).** That is real spread, and the
exploratory small-scale CPU run's own epoch-by-epoch log already showed val
AUROC bouncing non-monotonically (0.73 -> 0.65 -> 0.73 -> 0.77 across 4
epochs). A 4-epoch range in *where* patience triggers is consistent with
that same pattern here: patience is very plausibly triggering on val-AUROC
fluctuation rather than a clean plateau. That matters for how "seed noise"
should be read -- it is not purely weight-init/batch-order randomness, it
is confounded with noise in *which* epoch got selected as best. Not treated
as invalidating the result below, but it is a real caveat on what the
between-seed variance is actually measuring, not just how large it is.

**Correction #1 (this section was revised twice; both corrections below
were caught by review, not found independently -- recorded so the process
is visible, not just the final numbers).**

The first revision fixed the t-tests overstating things (multiple
comparisons) by promoting the sign test as the thing that "actually carries
the conclusion" at p=0.031, one-sided. That repeated the same mistake one
level up: p=0.031 does not clear the p<0.0167 bar this document itself set
for the family of 3 metrics either. Holding the sign test to a different,
looser standard than the t-tests -- after explicitly invoking that standard
to demote the t-tests two paragraphs earlier -- is not a fix, it is moving
the goalposts to keep a preferred conclusion. Applying the same bar
consistently: **nothing in this study clears p<0.0167 at n=5 seeds,
including the sign test.**

Worth stating precisely, because it is a fact about this design and not
about whether a real effect exists: the *best possible* one-sided sign-test
p-value at n=5 is (1/2)^5 = 0.03125, which is above 0.0167 regardless of
outcome. Five seeds cannot reach this corrected bar via the sign test even
with a perfect 5/5 result. Six seeds could: (1/2)^6 = 0.0156 < 0.0167. That
is a one-seed gap, not a large one, but it is real and this run does not
close it.

**Correction #2: the "~31 seeds" power calculation was circular.** Power
computed from the observed effect size is a monotone transform of the
p-value already reported -- it restates significance in different units
and adds no information beyond "this effect, if exactly this size, was
hard to detect at n=5," which is just p=0.32 said more elaborately. Removed.
Replaced with the confidence interval on the paired difference, which says
something the power number could not: what range of true effects is
consistent with the data, including effects in the wrong direction.

95% CI (t(4)=2.776, uncorrected) and Bonferroni-adjusted CI (98.33%,
t(4)=3.958, matching the p<0.0167 bar used above) on the paired difference,
all three metrics:

| metric | mean diff | 95% CI | Bonferroni-adjusted CI |
|---|---|---|---|
| AUROC | 0.0173 | [0.0018, 0.0328] | [-0.0048, 0.0394] |
| AUPRC | 0.0117 | [0.0007, 0.0227] | [-0.0040, 0.0274] |
| Sens@95%Spec | 0.0176 | [-0.0258, 0.0610] | [-0.0443, 0.0795] |

The uncorrected AUROC/AUPRC intervals exclude zero (consistent with their
uncorrected p<0.05); once corrected for testing 3 metrics, both intervals
*include* zero -- the same fact as "neither t-test clears p<0.0167,"
expressed as a range instead of a threshold, and a more honest one to look
at: even AUROC's data are consistent with anything from a small effect in
the wrong direction (-0.005) to more than double what was observed
(+0.039). Sens@95%Spec's interval is wider than the effect itself in both
directions -- consistent with a moderate effect either way, or none.

**Sample size, done properly this time: chosen in advance, not from what
was observed.** Picking a floor for "smallest effect that would matter in
practice" is a judgment call and is stated as one: 0.02 AUROC/AUPRC (a
common rough convention for the smallest AUROC/AUPRC gap treated as
practically distinguishable across pipeline comparisons in applied ML --
below that, most published benchmarks would not treat two numbers as
meaningfully different) and 0.05 for Sens@95%Spec (clinical screening
discussions of sensitivity tend to move in ~5-point increments; a 1-2 point
shift is not what would change a triage decision). These are defaults, not
measurements -- override them if domain judgement says otherwise. Using
those targets with the *observed* noise (sd of paired differences, which is
a variance estimate, not an effect-size estimate, so using it here is not
circular) at 80% power / alpha=0.05 two-sided:

| metric | target effect (chosen) | observed sd | seeds needed |
|---|---|---|---|
| AUROC | 0.02 | 0.0125 | 4 |
| AUPRC | 0.02 | 0.0089 | 2 |
| Sens@95%Spec | 0.05 | 0.0349 | 4 |

**Correction to how this table should be read (caught on a later pass):**
"5 seeds is sufficient (4 needed)" is not the right conclusion to draw from
it, and saying so was misleading. n=4 is how many seeds this design would
need to reliably detect an effect *if the true effect were as large as the
chosen practical floor (0.02 AUROC/AUPRC)*. It is a statement about the
design's power against a threshold, not a statement about what was found.
What was actually observed -- 0.0173 AUROC, 0.0117 AUPRC -- is *smaller*
than that floor in both cases. The correct statement is: **this study was
adequately powered to detect an effect large enough to matter by the
standard stated above, and the effect it actually measured is smaller than
that standard.** That is weaker than "the study found a practically
meaningful effect with room to spare," and it is a different claim than the
power number alone conveys -- the power number describes the ruler, not
what got measured with it. (The earlier, now-removed "~31 seeds" figure
made the opposite-flavoured mistake: it solved for detecting the exact
observed effect, a target nobody would choose in advance.)

**Net honest read, corrected twice now:** nothing in this study clears a
consistently-applied Bonferroni bar at n=5, including the sign test that
the first correction leaned on -- that was this document moving the
goalposts, caught on the second pass, not found independently. The
Bonferroni-adjusted confidence intervals for AUROC and AUPRC include zero;
the uncorrected ones do not, and all 5/5 seeds agree on direction, which is
suggestive but not, by this document's own stated standard, sufficient.
Read plainly: this run is *consistent with* leakage inflating AUROC/AUPRC
by a small amount, and *cannot rule out* zero or even a small effect in the
wrong direction once corrected -- and for Sens@95%Spec, the data are
consistent with a moderate effect in either direction. That is the honest
state of n=5. More seeds (6 for the sign test to even structurally reach
the corrected bar; the study is already adequately powered for the
practical-effect thresholds above) would narrow this; nothing here should
be written up as a settled result at this seed count.

## First A/B training run (small, CPU, exploratory -- not the headline number)

Ran train.py + experiment.py for the first time: arm A, arm B (seed 42),
and arm B again (seed 777), CPU only (CUDA install still pending on this
box), `subsample_n=800`, `epochs=4`, `image_size=224`, `crop_size=192`.
Timed one epoch first (42.4s) and estimated ~8.5 min total for all three
runs before committing to the full run -- actual wall time was close to
that estimate. Full numbers: `artifacts/results_table.md`.

| run | AUROC | AUPRC | Sens@95%Spec |
|---|---|---|---|
| A (image_random) | 0.745 | 0.803 | 0.349 |
| B seed 42 (patient_group) | 0.647 | 0.719 | 0.271 |
| B seed 777 (patient_group) | 0.701 | 0.759 | 0.294 |

A-vs-B gap (seed 42): 0.098 AUROC. B-vs-B seed spread: 0.054 AUROC. The
gap is bigger than the seed noise (~1.8x), and in the direction the
leakage hypothesis predicts (image_random inflated) -- but seed noise is
still more than half the size of the effect at n=800. Read plainly: this
run is consistent with the hypothesis, not proof of it at this scale.
Confusion matrices make the instability concrete too -- the same arm B
model swung from specificity-favoring (tn=66,fp=9) to sensitivity-favoring
(tn=36,fp=39) between the two seeds, a bigger behavioral swing than the
AUROC numbers alone suggest. Do not treat this run's numbers as the
project's headline result; re-run at full scale (no subsampling, more
seeds, real CUDA once available) before writing anything into the README's
headline table.

## Label strategy default switched: normal_column -> keywords

Investigated whether the patient-level `N` one-hot flag (`normal_column`)
or the per-eye diagnostic keyword string (`keywords`) is the more faithful
binary label, prompted by the earlier finding that `target`/`labels` vary
per-eye for 888/3034 multi-eye patients despite N/D/G/.../O never varying.

Real full_df.csv (6392 rows), reusing the actual adapter's
`_derive_eye_column`/`_label_from_keyword_string`:

- Pairwise agreement: normal_column vs keywords 87.9%, normal_column vs
  target 87.9%, **keywords vs target 100.0%** (every one of 6392 rows).
- Of the 675 patients where a target-derived binary label differs across
  eyes, keywords also differs for 674 (99.9%) -- and the reverse holds too
  (674/674, 100%). Two independently-authored eye-level signals agreeing
  this precisely is about as strong as evidence gets that both are
  measuring the same real eye-level ground truth, not encoding noise.
- Eyeballed 10 sampled normal_column-vs-keywords disagreements: all 10 are
  the same failure mode -- a unilaterally-diseased patient (patient-level
  N=0) whose *other* eye has the pathology, while this eye's own keyword
  string literally reads "normal fundus" and target agrees. normal_column
  mislabels every one of these healthy fellow-eyes as abnormal.

Conclusion: `keywords` is the better-supported eye-level label; switched
`configs/default.yaml`'s `label.strategy` default from `normal_column` to
`keywords`. `normal_column` stays implemented and selectable -- it's kept
as a comparison arm, not because it's a live candidate for correctness.

This also changes overall class balance: normal_column gave 2101/6392
(32.9%) normal; keywords gives 2874/6392 (45.0%) normal. That shift is
itself a data-curation decision with a measurable effect on the headline
numbers -- worth calling out plainly in the eventual writeup as exactly
the kind of choice this project exists to make visible, not just the
split strategy.

## Do not horizontally flip fundus images

Left/right eye is a real label here (bilateral disease correlation is the
whole premise of the patient-level split), and disc position relative to
the macula is a genuine anatomical cue a model can legitimately learn from.
A horizontal flip turns a left eye into an (anatomically wrong) right eye,
quietly corrupting that cue and, worse, is a leakage-adjacent trick that
would inflate scores by handing the model something it shouldn't be able
to infer that way. Light augmentation (small rotation, no flip) only.

## CUDA wheel pin (scripts/setup_env.ps1) is hardware-specific to this box

Dev GPU is a GTX 1050 -- Pascal, compute capability 6.1 ("sm_61"). PyTorch's
build matrix confirms Maxwell/Pascal/Volta were dropped entirely in the CUDA
13.0 wheels (Turing / sm_75 is the new floor), and Maxwell/Pascal were
*already* gone from the 12.8 and 12.9 wheel builds before that. So cu128,
cu129 and cu130 are not merely "newer" for this machine -- they ship no
sm_61 kernels at all, and would fail at first kernel launch regardless of
`torch.cuda.is_available()` reporting True. This machine's driver (576.88)
also caps out at CUDA 12.9 anyway, which would be a mismatch on its own even
against a GPU new enough to use those tags.

`setup_env.ps1` therefore pins `-TorchCudaTag cu126` by default -- the
newest tag that still ships Pascal support, matched with torch 2.14.0 /
torchvision 0.29.0 (verified against download.pytorch.org on 2026-09-05).

**This pin is specific to this machine, not a project requirement.** On a
Turing-or-newer GPU (RTX 20-series+) with a driver that supports it,
cu128/cu130/whatever is current is fine and preferable -- just override
`-TorchCudaTag` rather than assuming the default here is universal.
`scripts/doctor.py` checks for exactly this class of mismatch (sm_61
presence, and installed-wheel CUDA version vs. driver ceiling) so it gets
caught at setup time instead of at first kernel launch.

## Item 4: reports and figures regenerated with arm E and the hierarchy

Deliberate pass, per the same discipline as the earlier "Follow-up A"
regen: every chart labeled by run/split mode, nothing silently redefined.

**Two new charts** in `notebooks/findings_charts.py` (`fig_split_hierarchy`,
`fig_leave_one_site_out`), wired into `build_sections()` so they appear in
both `artifacts/findings_report.html` and (via `report.py`'s reuse)
`artifacts/qc_report.html`: the three-rung hierarchy as one AUROC bar chart
(A/B/E, each labeled by arm and split mode, captioned with the prevalence
caveat and the prevalence-matching result that rules it out as the
explanation), and the leave-one-site-out sweep with 95% CIs on every bar
-- site_4's positive exception shown at the same visual weight as the
other four, not hidden or asterisked away. Same two charts, restyled for
README-width reading, added to `notebooks/readme_figures.py` and embedded
in README.md's "Rungs 2 and 3" section.

**A real sign bug caught before publishing, not after**: the first draft
of `fig_split_hierarchy` computed the A-vs-B gap as B-A (matching the E-vs-B
convention used everywhere else in this project, copy-pasted without
checking) instead of A-B, the sign this project has used consistently
since the original A-vs-B result (README, WALKTHROUGH.md, docs/notes.md
all report it as "+0.0173" etc., leakage inflating the naive split's
score). Caught by actually reading the rendered takeaway text before
shipping it ("-0.018" read backwards from every other mention of this
number in the project) rather than trusting that green ruff/pytest meant
the chart was right. Fixed, and both `(A-B)`/`(E-B)` labels added inline
to the takeaway text so the convention is unambiguous without cross-
referencing this note.

**`dedupe.py` refactored, not just reused**: `embedding_duplicates`'s
inline feature-extraction code is now a standalone
`compute_resnet18_embeddings`, added for `notebooks/site4_followup.py`'s
distance-from-training check (Follow-up D) so it reuses the *exact*
extraction dedupe.py uses rather than a re-derivation that could drift.
`findings_charts.py`'s own `_compute_embeddings` (a near-identical copy,
predating this refactor) was deleted in favour of importing the same
function -- one embedding-extraction implementation in the whole project
now, not three.

**A real, if minor, infrastructure gap found while trying to update arm
E's official cohort, and deliberately not patched around**: `config_hash`
(`utils.py`) hashes the resolved config dict only -- it has no way to
detect that `artifacts/site_labels.parquet` / `site_group.json`'s
*content* changed (the patient-consistency fix, Follow-up C) without any
config value changing. Retraining arm E under the fixed split and
recording it via the normal `append_run_index` path would therefore get
the *same* config_hash as the old pre-fix cohort, and
`load_current_run_metrics`'s cohort-selection logic -- designed exactly to
stop incompatible cohorts blending -- would silently average all 10 runs
(5 old + 5 new) together instead of picking one. Not fixed here (out of
scope for a reports-and-figures pass, and the fix's own effect is <0.002
AUROC, test fold unchanged, per Follow-up C) -- worked around for these
specific charts by sourcing arm E's numbers as literals from
`arm_e_robustness_checks.py`'s already-verified retrain output (the same
pattern this file already uses for `FIRST_RUN_*_AUROC`), with a comment
at the point of use explaining exactly why. Flagged here as a real gap in
`config_hash` for anyone who next needs to update a split file's content
without changing its config -- the honest fix is hashing the split/label
artifacts' own content into the hash, not touched in this pass.

**Landing page (`docs/index.html`) and clinical summary
(`docs/summary.md`) rewritten**, not just appended to: both predated arm E
entirely and materially understated the project's own headline finding --
`docs/index.html`'s intro described the leakage effect as "modest,
non-replicating" (true of the patient-level effect alone, false of the
project's actual current headline), and `docs/summary.md`'s "Bottom line"
told a clinical reader the leakage problem was "a dial, not a switch" with
no mention that a much larger, mostly-site-general effect exists. Both
now lead with the three-rung hierarchy and the site-holdout result,
without deleting the original patient-level findings (still true, just no
longer the whole story) -- same "keep both, cite exact numbers" discipline
as every other update in this file.
