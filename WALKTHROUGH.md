# Walkthrough

> Written for a reader who has to defend this project in a technical interview.
> **Read this file before reading the code.**

## 1. The claim

The claim, stated as the hierarchy it turned out to be rather than the
single number the project started with: **splitting by image leaks
patients; splitting by patient leaks sites instead; only splitting by
site isolates the disease signal.** Each level of grouping is
necessary and, on its own, insufficient for the level above it. All
three rungs are measured directly in this project, not asserted:

1. **Image-level splitting leaks patients** — 42.0% of patients cross
   a naive train/test boundary (§1 below, exact count).
2. **Patient-level splitting leaks sites instead** — a classifier
   recovers which clinic captured a *preprocessed* photo 84% of the
   time, and site is entangled with diagnosis at p≈1e-16 (§5 and the
   domain-shift audit in §9).
3. **Site-level splitting (arm E) isolates the signal, at a measured
   cost**: AUROC drops over 3x the size of the original patient-level
   effect when entire sites are actually held out, the cleanest,
   Bonferroni-surviving result in this project (§9).

The rest of this section works through rung one in the detail it was
originally measured in; rungs two and three get the same treatment in
§9, once the machinery (splitting, curation, the experiment arms) that
rung one motivated has been introduced. Read as one arc, not three
separate findings: the project's central result is the hierarchy, not
any single rung of it.

Splitting a fundus dataset by image instead of by patient inflates measured
model performance, because bilateral disease correlation and repeat-visit
recapture mean the "unseen" test set is not actually unseen. This project
measures that inflation directly rather than asserting it: the same ResNet18,
the same hyperparameters, the same training-set size, trained once on an
image-random split (arm A) and once on a patient-grouped split (arm B),
5 seeds each, on the real ODIR-5K dataset (6392 images, 3358 patients).

The exact-count evidence needs no statistics: **42.0% of patients (1412/3358)
land on both sides of the image-random split's folds.** Patient-grouped
splitting eliminates this outright — 0 patients cross a fold boundary, by
construction, every time.

The downstream cost is real but modest, and this project's most honest
result about it is that **it did not replicate cleanly on a second run**.
The first full run gave a paired mean AUROC gap of 0.0173 (image-random
scoring higher), consistent across all 5/5 seeds, with an uncorrected 95%
CI excluding zero but not surviving Bonferroni correction. A second run —
same nominal seeds, training-set size matched within 0.85% of the first
(the difference comes from also matching against arms C/D, which didn't
exist yet the first time) — gave +0.0079, not statistically significant,
3/5 seeds agreeing on direction, not 5/5. Both numbers are reported, not
just the first, more favourable one. The full statistical account —
including two rounds of self-correction where an earlier draft of the
first run's analysis overstated the evidence, *and* the failure to
replicate on the second — is in `docs/notes.md` and §9. That process is
the claim as much as any single number is: a project whose central result
survived being checked twice and being re-run once, and changed both
times, is more trustworthy than one that got a clean answer on the first
pass and stopped looking. What *does* replicate, exactly, every time,
because it's a count rather than a statistic: the 42.0% patient overlap
above.

Investigating *why* the effect is smaller than 42.0% overlap would suggest
turned out to be the most informative part of the project: fellow eyes
share the same binary label only 77.8% of the time (vs. a 50.5% chance
floor at this dataset's class balance) — leaking a patient's identity does
not mean leaking their label, it means leaking a noisy ~78%-reliable hint
about it. See §5.

## 2. Dataset and label derivation

**Why ODIR-5K**: it is organised by patient with a left and right fundus
image per patient, which is exactly the structure that makes patient-level
leakage measurable and quantifiable rather than theoretical. Most public
fundus datasets (e.g. single-image classification sets like APTOS) don't
have this bilateral structure, so the leakage question can't even be posed
the same way.

**A real assumption failure, corrected against the actual data**: the
adapter was first written assuming `full_df.csv` was *wide* (one row per
patient, `Left-Fundus`/`Right-Fundus` columns). The real file is *long*
(one row per eye — confirmed: 6392 rows, 3358 unique IDs, 6392 unique
filenames). The wide-format code would have silently melted every row into
two, doubling the dataset without erroring. Caught by checking the actual
downloaded file's shape before trusting the adapter, not by an error at
runtime — nothing in a wide-format melt of a long-format file would have
raised an exception, it would have just quietly produced a wrong but
plausible-looking manifest. `eye` is derived from the filename's own
`_left`/`_right` suffix (asserted, fails loudly on any filename that
doesn't match), not from the `Left-Fundus`/`Right-Fundus` columns, because
those columns are populated for both sides even when only one side's
image/row actually exists — a real quirk of the source data (324/3358
patients, 9.6%, have only one usable eye).

**Label derivation — investigated, not assumed, and it mattered**: ODIR
ships two candidate label sources. `normal_column` (the patient-level `N`
one-hot flag) forces both eyes of a patient to the same label. `keywords`
parses the per-eye diagnostic keyword string. The two disagree on 12.1% of
rows. Rather than picking one by convention, checked which is better
supported: `keywords` agrees with the (also per-eye) `target` column on
100.0% of all 6392 rows, and of the 675 patients where `target` differs
across eyes, `keywords` also differs for 674 of them (99.9%) — two
independently-authored eye-level signals agreeing this precisely is strong
evidence both are measuring real per-eye ground truth. Eyeballing 10
sampled disagreements confirmed the mechanism: `normal_column` mislabels
the healthy fellow-eye of a unilaterally-diseased patient as abnormal,
every time, because it can only see the patient-level flag. Switched the
project default to `keywords`; `normal_column` stays implemented as an
explicit comparison. **This label-source choice is itself a data-curation
decision with a measurable effect on the headline numbers** (32.9% vs.
45.0% normal), before curation or splitting are even in play.

## 3. Manifest design

One canonical schema (`image_path`, `patient_id`, `eye`, `age`, `sex`,
`label`, `dataset_name`) that every adapter must produce, so adding a
second dataset means writing one function, not touching split/train/
experiment code. `patient_id` is the load-bearing column: it is the only
thing that determines whether a naive split leaks, so `validate_manifest`
enforces it's a non-null string on every row and every referenced
`image_path` actually exists on disk — failing loudly with every offending
row named, not silently dropping bad rows. Adapter code separately never
assumes exactly two rows per patient (single-eye patients are 9.6% of the
data); a hardcoded `2 *` assumption was caught and removed from an early
version of the test suite specifically because it would have silently
passed on data where it happened to be true and failed confusingly later.

## 4. Splitting

`image_random_split` (wrong, on purpose) is an ordinary stratified
`train_test_split` over individual images, blind to `patient_id`.
`patient_group_split` (right) uses `StratifiedGroupKFold`, holding out one
fold as test, then splitting the remainder again for val — so a patient's
images are guaranteed to land in exactly one fold. Perfect
stratification is impossible under a group constraint (you can't
simultaneously balance class *and* keep every group whole when group sizes
vary), so the achieved class balance per fold is logged and reported, not
assumed to hit the configured fraction exactly — on the real data it lands
within about half a percentage point in practice, close enough not to
matter, but that's an empirical finding, not a guarantee the code makes.

`patient_overlap` is the direct, assertable proof: it counts how many
patient IDs appear in more than one fold. On the real full dataset:
**0 for `patient_group`, 1412/3358 (42.0%) for `image_random`.** Every
`patient_group_split` result in this codebase is checked against this
function before being trusted for anything downstream.

## 5. Why leakage is worse in retinal imaging than elsewhere

Four compounding reasons, in decreasing order of how much this project can
actually quantify each one:

1. **Bilateral disease correlation.** Most systemic and many ocular
   conditions affect both eyes together (diabetic retinopathy, glaucoma,
   hypertensive changes). If a model has seen the left eye's presentation,
   it has seen strong evidence about the right eye's — not because it
   learned anything general, but because the two eyes are correlated by
   the same underlying disease process. **Measured directly on this
   dataset: fellow eyes share the same binary label 77.8% of the time,
   against a 50.5% chance floor given the class balance.** That 27.3
   percentage-point excess over chance is the quantitative size of this
   effect here.
2. **Two eyes, one patient ID.** This is what makes the leak mechanical
   and unavoidable under a naive split: a random shuffle of *images*
   doesn't know "these two rows are the same person," so it puts them on
   opposite sides of the boundary constantly (42.0% of patients, measured).
   A domain with one image per subject (e.g. many chest X-ray sets, one
   film per patient per visit) doesn't have this specific failure mode at
   all — leakage there comes from other sources (repeat visits, near-
   duplicate scans), not from an *inherent* two-samples-per-subject
   structure.
3. **Repeat visits and same-session recaptures.** A clinic re-photographs
   a bad capture, or the same patient returns for a follow-up. This
   project found a concrete instance of the *adjacent* problem — the same
   underlying photograph filed under two **different** patient IDs (8
   confirmed pairs, pixel-verified) — which patient-grouped splitting
   cannot catch, because it only groups by *declared* patient ID, and
   these are declared differently. Only deduplication closes that gap;
   splitting strategy and deduplication defend against genuinely different
   failure modes, not the same one twice.
4. **Task coarseness compounds all of the above.** Binary normal/abnormal
   collapses 8 original ODIR categories into one "abnormal" bucket, so two
   *different*, unrelated diseases in the two eyes both count as
   "abnormal" and therefore "concordant" here — likely *understating* true
   bilateral consistency relative to a per-disease task, where the same
   specific disease in both eyes would show even higher concordance. Not
   measured directly in this project (a multi-class or per-disease
   experiment was assessed as plausible but not run), but it's the
   direction any coarseness correction would point.

A domain where this is milder: single-image-per-subject datasets with no
bilateral structure and infrequent recapture (e.g. many single-photograph
dermatology sets) — the *image*-vs-*patient* distinction that dominates
this project's whole design collapses to "one row per subject," so a naive
random split is simply correct there, not merely less wrong.

## 6. Quality scoring

Four classical CV metrics (variance of Laplacian + Tenengrad for blur,
FOV-interior exposure stats, per-quadrant illumination CV), combined into a
transparent weighted sum (`gradability_score`), not a learned model — "why
was this image rejected" needs to be a one-sentence, inspectable answer,
which a hand-weighted sum gives for free and a learned classifier doesn't.

**Two documented traps, both real and both caught before trusting a
number, not just avoided in the abstract:**

- *Resolution dependence.* `variance_of_laplacian` is naturally larger for
  higher-resolution images regardless of true focus, so blur metrics must
  run on a fixed resize, not raw resolution. Checked directly rather than
  assumed moot: this project's own `preprocessed_images/` is uniformly
  512×512 (200-image sample, zero exceptions), so the trap is inert for
  what's actually scored here — but the raw `Training Images/` mixes 12+
  distinct resolutions in the first 50 files alone, so the fixed-resize
  step stays unconditional in the code, correct for that folder and any
  future dataset even though it's a no-op today.
- *FOV-interior exposure.* Computing exposure over the whole frame makes
  every image look underexposed, since the black background outside the
  circular retinal field dwarfs the actual content in pixel count.
  `exposure_metrics` takes the detected FOV mask explicitly and restricts
  every statistic to it.

**A threshold bug that would have rejected half the dataset, caught before
reporting the number**: the first FOV-validity check compared a *fitted
enclosing circle's* geometric extent against the frame. Since this
dataset's FOV is pre-cropped to fill the frame almost exactly (median
radius fraction 0.998), ordinary off-centre circle-fitting noise on a real
photograph triggered "clipped" on the majority of completely normal
images — 51.1% reject rate on the first run. Diagnosed rather than
reported: switched to checking whether the *thresholded contour itself*
touches the border (38.5%, still wrong) then to a contour-area/circle-area
circularity ratio, and found by direct visual inspection at several
threshold values that circularity conflates three different things on
this dataset — real truncation, complete-but-*oval*-shaped crops (a
legitimate, common variant here, not a defect), and hazy/low-contrast
images with a ragged threshold boundary. **Fix: `fov_clipped` is computed
and stored for visibility but no longer gates rejection**; only the
FOV-radius-fraction check (a real, uncontroversial "is the field too
small" signal) does. Final reject rate: **43/6392 = 0.7%**, verified
sane three ways — a contact sheet of the 20 lowest/20 highest scoring
images shows a clean visual gradient; all 43 rejects inspected by eye show
genuine degradation with no clear false rejects (some near-boundary cases
are honestly ambiguous, which is expected at any hard cutoff, not a bug);
and a 400-image sample of the *raw*, not-yet-curated `Training Images/`
scored with the identical code and thresholds rejects 4.2% — a 6x
relative difference confirming the low number reflects that this dataset
was already curated upstream, not that the module is lax.

**A known, unfixed limitation, stated plainly rather than left implied**:
uniform haze (e.g. dense cataract, severe media opacity) is not reliably
caught. Two essentially featureless, uniformly hazy images score 0.943 and
0.979 — near the top of the entire dataset. `illumination_uniformity`
measures *unevenness*, and uniform haze is by definition even, so it reads
as good; blur metrics stay above threshold because moderate haze doesn't
eliminate all high-frequency content (compression artifacts, faint
reflections still register). Classical per-pixel/per-quadrant statistics
of the kind used here structurally cannot distinguish "uniformly hazy"
from "uniformly clear" — a real deployment would need an added
contrast/entropy-style check or a learned gradability classifier
specifically for this failure mode.

## 7. Deduplication

Two methods, deliberately catching different things: `phash` (64-bit
perceptual hash) catches near-identical pixels — crops, rescales,
recompression. `embedding_duplicates` (pretrained ResNet18 penultimate
features, cosine similarity) catches the same eye photographed twice under
different lighting or exposure, where the raw pixels differ substantially
but the content doesn't.

**The pattern underneath both threshold failures, stated once here because
it explains both**: generic image-similarity methods — a 64-bit
perceptual hash, an ImageNet-pretrained embedding — are calibrated on the
assumption that "different photos" carry a certain amount of ordinary
inter-image variation. Retinal fundus photographs have far less of that
variation than the natural-image domains these methods were designed
around: every image is a dark circular field with broadly similar
framing, illumination, and colour palette, and — specific to this
modality — a *patient's own two eyes* are anatomically more similar to
each other than two random strangers' retinas are. Both thresholds
shipped with this project's config (`phash_hamming_max: 6`,
`embedding_cosine_min: 0.98`) were calibrated for the *general* case and
were both too loose for this specific one, in the same direction, for the
same underlying reason. This is the second time in this project a
standard method needed fundus-specific tuning (quality's FOV-circularity
check, §6, was the first) — worth naming as a pattern, not just two
unrelated bugs.

**Threshold validation, done honestly, twice over, in the same module**:

- *phash*: the configured default (`hamming<=6`) flags 13733 candidate
  pairs on the real 6392-image dataset. Visually inspected several — the
  overwhelming majority are false positives, because fundus photography
  shares enough generic macro-structure (dark background, circular field,
  similar framing) that a coarse hash collapses unrelated images together.
  **Fix: every phash candidate is verified by actual pixel difference
  before being trusted**, not just the strictest hamming==0 bucket an
  earlier ad hoc pass checked (which found only 2 genuine pairs) — full
  verification of all 13733 candidates found **8** genuine pairs, because
  2 real duplicates sat at hamming distance 1-6 and would have been missed
  by only trusting the tightest bucket.
- *embeddings*: the configured default (`cosine>=0.98`) produced 165
  candidates on the first real run, almost all logged as cross-patient
  matches — a volume as suspicious as the phash false-positive rate, so
  checked before trusting it. Pixel-difference verification doesn't apply
  here (embeddings are explicitly meant to catch pixel-*different*
  content), so verified by eye instead, and found the threshold **also**
  caught 4 genuine fellow-eye pairs as false "duplicates" — exactly the
  risk flagged as worth checking in advance. Raised to `cosine>=0.99`:
  zero fellow-eye false positives, 10 candidates, all confirmed by low
  pixel-difference and (for the 3 not already found by phash) direct
  visual inspection — unmistakably the same eye in each case.

**Empirical justification for 0.99, not just "raised until the false
positives went away"**: computed cosine similarity for a 400-patient
sample of genuine fellow-eye pairs and for all 8 confirmed duplicate
pairs (11 pair-instances). Fellow eyes: mean 0.933, max 0.975, 99.5th
percentile 0.973 — confirming directly that a patient's own two eyes really
do sit close to where genuine duplicates live in this embedding space, not
just plausibly. Confirmed duplicates: mean 0.993, min 0.962. **0.99 sits
just above the fellow-eye ceiling with real margin for 10 of the 11
duplicate-pair-instances — but not for all of them.** One confirmed
duplicate, 3297 vs 4542 (pixel-diff verified via phash), has cosine
similarity 0.962 — *below* the fellow-eye sample's own maximum. No
embedding threshold at any value would catch that pair without also
catching genuine fellow eyes; it was found only because phash's
independent, pixel-structure-based signal doesn't care that this
particular duplicate's two copies happen to differ enough in surface
appearance (lighting/colour grading) to read as dissimilar to a semantic
embedding. **This is the concrete, not-just-theoretical reason two methods
are needed**: of the 11 duplicate-pair-instances, phash alone would have
found 8 (missing the 3 that only embeddings caught); embeddings alone
would have found 10, permanently missing the one pair discussed above
(3297 vs 4542, cosine 0.962). They fail in
different directions on this modality — phash over-triggers on shared
macro-structure, embeddings under-triggers on genuine duplicates whose
lighting diverged enough — and neither failure mode is visible from
inside the other method.

**Final, real dataset-integrity finding**: 11 distinct duplicate clusters,
22 images, across 8 unique cross-patient pairs — the same underlying
capture filed under two *different* patient IDs, confirmed by near-zero
pixel difference on every pair. **The money metric**: of the 11 unique
verified pairs (8 phash + 10 embedding, 7 found by both), 3 straddle the
`image_random` split's folds and 2 straddle `patient_group`'s — materially
the same order of magnitude for both, which is the point: patient-grouped
splitting has no mechanism to catch a duplicate filed under two different
patient IDs, since it only keeps a single declared ID's images together.
This class of leakage is deduplication's job specifically, not
splitting's — the two stages defend against genuinely different failure
modes.

## 8. Model and metrics

ResNet18, pretrained, new 2-class head, CrossEntropyLoss, Adam — the
project's own thesis is that the data path is the contribution, so the
model is kept deliberately boring and identical across every arm; a
fancier architecture would only add a second, uncontrolled variable to a
comparison whose entire value is having exactly one.

**Why not accuracy**: at this dataset's ~45%/55% class balance, accuracy
is not badly imbalanced-sensitive here, but AUROC/AUPRC still say more —
they're threshold-independent, so they're not sensitive to *where* the
model happens to put its decision boundary, only to whether it ranks
positives above negatives correctly. **Sensitivity at 95% specificity is
the clinically meaningful addition**: a screening tool's real operating
point is chosen to keep the false-positive (unnecessary referral) rate at
a level clinics can absorb, then sensitivity at that fixed point is what
determines how many real disease cases get missed. AUROC alone can be
identical between two models with very different sensitivity at the
specific operating point that would actually be deployed.

No horizontal flip in augmentation, only small rotation — a flip turns a
left eye into an (anatomically wrong) right eye, and disc position
relative to the macula is a real anatomical cue a model can legitimately
learn from; flipping silently corrupts that cue and is a
leakage-adjacent trick (it would let the model exploit fellow-eye mirror
symmetry in a way that doesn't correspond to real generalisation).

## 9. Experiment design

Same seed pattern (`[cfg.seed, cfg.seed+1, ..., cfg.seed+n_seeds-1]`),
same hyperparameters, and — the part that mattered most in practice —
**training-set size matched across every arm being compared, computed
fresh from all four arms' actual curated pools, not assumed**. Curation
removes images (quality: 43/6392, 0.7%; quality+dedupe: an additional 11
images/6349, 0.2%), so C and D's natural pools are smaller than A/B's raw
one; capping only within "raw" arms while leaving curated arms uncapped
would let a curated arm's smaller pool masquerade as a curation effect
when it's actually just a dataset-size effect. Natural train sizes on the
real data: A=4473, B=4474, C=4443, D=4435 — cap=4435, so even A/B (which
needed no curation) lose a fraction of a percent to match. Small in this
case, but the mechanism is what matters, not the magnitude: **if arm C had
beaten arm B without this cap, the honest reading would be "C had less
data variance from a smaller, easier subset," not "curation helped."**

What is controlled: model, hyperparameters, seed sequence, train-set size,
split strategy (patient_group for B/C/D). What is *not* controlled and
would confound the result if ignored: image resolution/preprocessing
pipeline differences between real deployment data and this benchmark
(everything here comes from one already-curated Kaggle release); the
specific quality/dedupe thresholds, which were investigated and corrected
during this project but are not claimed to be optimal, only evidence-based
and stated as such.

**Prediction, stated in `docs/notes.md` before arms C and D were run, not
after**: quality curation removes 43 images (0.7%) and deduplication
removes at most 22 (0.3%, and only from whichever side of each pair
curation decides to drop). Neither is large enough to plausibly move an
AUROC measured against a paired seed-to-seed noise floor of ~0.05 (the A/B
result). **Arms C and D were predicted to be statistically
indistinguishable from arm B.**

**Result — mixed, and reported that way rather than rounded to a single
verdict**: mean AUROC across 5 seeds — A 0.8014, B 0.7936, C 0.7790,
D 0.7820. Full paired-comparison detail (Bonferroni-adjusted CIs, sign
consistency) is in `docs/notes.md`; the headline of it:

- **D vs B: the null prediction held.** No metric significant, corrected
  or not; mixed signs on AUROC/AUPRC.
- **C vs B: the prediction did *not* fully hold — this is the surprising
  result, and it's the one worth reading closely, not the confirming
  one.** AUROC is lower for C than B by a small amount (-0.0146),
  consistently across all 5/5 seeds, surviving Bonferroni correction
  across the 6-test family (p=0.0078 against a 0.00833 bar) — though the
  corrected CI's upper bound sits at -0.0002, a hair under zero, so this
  is reported as a real but marginal finding, not a decisive one.
  **Why, investigated with three hypotheses rather than left at the first
  plausible-sounding one:** (1) *quality correlates with disease* —
  tested and refuted in the opposite direction: the 43 rejects are 62.8%
  normal against a 45.0% baseline (rejects skew normal, not abnormal;
  p=0.019), and across all 6392 images abnormal examples score very
  slightly *higher* quality on average, not lower (p=0.004). (2)
  *composition, not size* — tested and confirmed, much larger than
  expected: B's and C's actual training sets, both exactly 4435 images,
  share only 69.2% of those images, because `patient_group_split` is
  recomputed fresh per arm and `StratifiedGroupKFold` reassigns a large
  fraction of fold membership from a small pool change. (3) *magnitude* —
  removing 43 random, quality-uncorrelated images and re-splitting
  produces 69.4% overlap with B's original training set, statistically
  indistinguishable from the real 69.2%: the actual perturbation this
  comparison measures is ~31% of the training set, not the <1% the raw
  removal count implies. **Revised leading explanation**: split-algorithm
  sensitivity to input perturbation, not curation removing informative
  content — the original "informative-but-hard-to-grade" hypothesis was
  the flattering account and didn't survive being checked. Full detail:
  `docs/notes.md`.
- **The A-vs-B gap itself did not replicate on this run.** Same nominal
  seeds, matched size within 0.85% of the original A/B-only run — and
  this time the AUROC gap was +0.0079 (not +0.0173), p=0.29 (not 0.037),
  3/5 seeds agreeing on direction (not 5/5). This is not a contradiction
  to explain away; it is the underpowered-at-n=5 caveat from §1, now
  demonstrated by an actual second draw rather than argued from a power
  calculation alone. If a ~38-image (0.85%) difference in the training
  pool — an artefact of matching against arms that didn't exist in the
  first run — was enough to flip "significant, unanimous" to "not
  significant, 3/5," n=5 was always going to be this fragile.

Curation earning its place in this pipeline by catching a real integrity
problem (8 cross-patient duplicate pairs, §7) rather than by improving a
metric turned out to be literally the honest description for arm D, and
worth stating just as plainly for arm C, which *did* move a metric — just
not in the direction curation is usually assumed to move it, and not by
much. Full numbers, all four arms, both split strategies where
applicable: `docs/notes.md` and `artifacts/results_table.md`.

**Everything above this line describes results produced before a fix
implemented in a later session — kept in full, not deleted, because it's
the evidence the fix was built to respond to.** The next section reports
what changed and why the numbers above are labelled "before-fix."

### The split-then-curate fix: diagnosing a confound, then eliminating it

The composition finding above (B and C's training sets sharing only
69.2% of images despite matched size) was a diagnosis, not a fix — the
underlying defect (`patient_group_split` recomputed fresh per arm,
letting `StratifiedGroupKFold` reshuffle a large fraction of fold
membership from any small pool change) stayed in the code, flagged in
`docs/notes.md` as a design lesson for future work.

**The fix**: split once on the raw pool (already what `retinaprep
split` persists to disk), then for curated arms *filter* that fixed
split down to whichever images survive curation, instead of
recomputing. Every surviving image keeps the exact fold it was
originally assigned; the only variable between arms becomes which
images were removed, not how the remainder got reshuffled.
`experiment.split_mode: persisted_base` is now the default (the old
`recompute_per_arm` behaviour is kept, fully working, and selectable —
the numbers above were produced with it and must stay reproducible on
demand). One direct, deliberate consequence: **arm sizes are no longer
matched to a common cap.** Matching size was exactly what forced the
fresh per-arm recompute in the first place; under the fix, the size
difference between arms *is* the treatment (how much curation actually
removed), not a confound to correct for.

**Verified before spending any GPU time on it**: under the fix, C's and
D's training sets are provably subsets of B's — 100% overlap (as a
fraction of the smaller, curated set), not 69.2%. This is a pure
split/curation computation, no training involved, and it's the direct
check that the fix does what it claims.

**Prediction, stated in writing before re-running anything**: with ~43
images removed out of ~4470 and fold membership now stable, C-vs-B
should show a much smaller effect than the -0.0146 previously measured,
because most of that original effect was resampling, not curation.

**Re-ran all four arms, 5 seeds, full dataset, GPU, under the fix, at
natural (uncapped) sizes.** Mean ± std across 5 seeds:

| Arm | N train | AUROC | AUPRC | Sens@95%Spec |
|---|---|---|---|---|
| A (image_random) | 4473 | 0.8098 ± 0.0069 | 0.8567 ± 0.0052 | 0.4554 ± 0.0262 |
| B (patient_group) | 4474 | 0.7915 ± 0.0099 | 0.8439 ± 0.0104 | 0.4327 ± 0.0304 |
| C (quality-curated) | 4446 | 0.7944 ± 0.0094 | 0.8462 ± 0.0076 | 0.4020 ± 0.0142 |
| D (quality+dedupe) | 4436 | 0.7856 ± 0.0135 | 0.8395 ± 0.0083 | 0.4191 ± 0.0158 |

**The prediction held, on the identical statistical bar used above**
(Bonferroni across the same 6-test family, alpha=0.00833):

- **C vs B, AUROC: -0.0146 → +0.0029.** The one comparison in this
  entire project that survived Bonferroni correction has disappeared
  and flipped direction (p=0.0078, sign 5/5, before; p=0.6160, sign
  3/5, after). AUPRC moves the same way (-0.0068 → +0.0023). This is
  the significant result, not a disappointing one: the project
  diagnosed a methodological artifact from indirect evidence (the
  69.2%/69.4% overlap numbers) and predicted in writing what
  eliminating it should do to the headline comparison, then it did
  that.
- **D vs B: null both before and after** — consistent with dedupe
  removing too few images to plausibly move a metric either way,
  regardless of split-recompute behaviour.
- **One new, honest, unconfirmed lead**: Sens@95%Spec now shows a
  sign-consistent (5/5) *decrease* for C vs B (-0.0307, p=0.0384
  uncorrected) that does not survive Bonferroni correction. Reported as
  exactly that — an uncorrected, sign-consistent signal worth more
  seeds, not a finding — for the same reason the variance-instability
  lead in the next section wasn't promoted either.

Full per-seed values, the complete Bonferroni-adjusted CI table, and the
A-vs-B third measurement this run also produced (a methodological
caveat applies — see `docs/notes.md`) are in `docs/notes.md`, "Split-
then-curate ordering fixed."

### Is the naive split unstable, not just optimistic?

A second-sounding, independent argument for grouped splitting suggested
itself from the second run's numbers: arm A's AUROC std is 0.0129 against
arm B's 0.0028 — 4.6x — which would mean the naive split isn't just
optimistic on average, its score also depends on *which* patients
happened to straddle the fold boundary, making it less run-to-run stable
too. Checked against the first A/B-only run before reporting it as a
second confirmed finding, exactly the way the mean-difference finding was
checked (§1) — **it does not hold up the same way in both runs.** In the
first run, A was actually *less* variable than B on AUROC (0.0069 vs
0.0083) and AUPRC — the opposite direction — and only sensitivity at 95%
specificity showed A more variable there (2.58x), which is the metric
showing the *weakest* version of the pattern in the second run (1.43x).
No individual run's variance difference reaches conventional significance
(Levene's test, smallest p=0.081); an informally pooled estimate across
both runs (ratio ~1.8–2x for AUROC/AUPRC, treating the two runs' 5 seeds
each as 10, with the caveat that the runs used slightly different
training-set caps) is directionally suggestive but still not significant
(p=0.08–0.20) at this sample size.

**Reported honestly rather than promoted to a second headline finding**:
this does not clear the bar every other claim in this project has been
held to. It was hoped this would be "a result that replicated when the
headline didn't" — checking it against the first run the same way the
mean-difference was checked shows that's not an accurate description of
what the data supports. It's a suggestive, unconfirmed lead worth more
seeds to resolve, reported as exactly that, not inflated into a
second, independent argument for grouped splitting just because the
second run's numbers looked clean on their own. This is the same
discipline applied to itself: a hypothesis this project wanted to be true
doesn't get a pass on the checking that every other claim here got.

### Reproducibility: why runs no longer overwrite each other

`train.py` used to write every run to a fixed path,
`artifacts/runs/<arm>_seed<seed>/metrics.json`. That is exactly how the
*first* full-scale A/B run's per-seed data was lost: a later run reused
the same arm/seed names (`A_seed42`, `B_seed43`, ...) and silently
overwrote the earlier files. The two runs' *aggregate* numbers both made
it into README.md/docs/notes.md before that happened, but the first run's
raw per-seed JSON is gone — it now has to be cited from this document's
own table rather than regenerated from disk, which is exactly the
failure mode a reproducibility-focused project should not have.

Fix: every run now writes to a unique directory —
`<run_name>_<UTC timestamp>_<8-char sha256 of the resolved config>` — and
appends one entry to `artifacts/runs/index.json` (arm, seed, config hash,
timestamp, run directory, key metrics). Nothing is ever overwritten
again. The remaining design question this raises: once two cohorts for
the same arm can coexist (an old config's runs and a new config's runs),
naively averaging every run ever found under an arm would silently blend
incompatible cohorts into one meaningless number. `retinaprep.utils.
load_current_run_metrics` resolves this by selecting, per arm, only the
runs sharing that arm's *most recently recorded* config hash — older
cohorts stay fully on disk and in `index.json` for provenance (and for
exactly this kind of postmortem), they just drop out of the current
headline aggregation. `report.py`, `experiment.py`'s results table, and
`findings_charts.py`'s arm-results chart all read through this one
function now, rather than three separate ad hoc directory scans that
would each need to re-implement the same cohort logic (and would drift
from each other if they didn't).

The same investigation surfaced a second, unrelated reproducibility gap
worth recording here rather than only in `docs/notes.md`:
`configs/default.yaml` defaulted to `dataset.subsample_n: 2000` and
`experiment.n_seeds: 1`, neither of which was ever the config actually
used for a reported headline number (both used the full dataset and 5
seeds, via CLI overrides that were never written back into the checked-in
default). A clean checkout running the documented Quickstart commands
with no overrides would therefore have silently reproduced a different,
much smaller experiment than the one this document describes, with no
error to flag the mismatch. The defaults now match what was actually
measured; a fast local smoke test is an explicit opt-in override, not the
silent default.

### Cross-camera domain-shift audit: site is recoverable, and it's entangled with diagnosis

Roadmap item 2. ODIR-5K mixes Canon, Zeiss and Kowa across several
Chinese centres with no explicit camera/site column — raw image
resolution (before this project's preprocessing resizes everything to a
common 512x512) stands in as a proxy. Stated once, applies throughout:
**this is a proxy for camera, not the camera itself** — if several
cameras share a resolution, it under-counts real sites.

Matched all 6392 images against the raw `Training Images/` folder,
found 97 distinct resolutions, and DBSCAN-clustered them (eps=100px) into
43 raw groups; groups under 50 images (long-tail/anomalous resolutions)
were consolidated into one "Other" bucket — **20 final site classes**.
Checked directly: 99.0% of two-eye patients share an identical raw
resolution across both eyes, confirming this is overwhelmingly a
per-patient property.

**The test that matters**: trained a fresh ResNet18 on the *already-
resized* 512x512 images — not the raw ones — to predict which of the 20
site classes an image came from, on a patient-grouped split so a
patient's fellow eye couldn't hand it a trivial shortcut. **Test
accuracy 0.8397 against a 0.2932 majority baseline.** Because every
image the model saw was already resized to the same dimensions, this
result can't be about pixel dimensions — the signal is in the optics or
colour rendition, and it survives the exact preprocessing step this
project's whole pipeline runs on.

**Then the question that actually matters for this project's leakage
claims**: does site correlate with diagnosis? Patient-level chi-square,
site (20 classes) against the genuinely patient-level `N` flag (not the
per-eye `label` column this project's main task trains on, which
disagrees across a patient's own eyes 22.2% of the time — see §2 — so
picking one eye's value for a patient-level question would be
arbitrary): **chi2=115.18, p=8.8e-16.** Broken down by individual
category, every one of diabetic retinopathy, hypertensive retinopathy,
glaucoma, cataract, myopia, and "other" shows a highly significant site
correlation (all p<0.005); only age-related macular degeneration does
not (p=0.12) — itself a specific, checkable exception worth a direct
look later, not a gap glossed over.

**What this means, stated plainly**: patient-level splitting (this
project's entire design) stops a model from memorising one *patient's*
fellow eye across train/test, but does nothing to stop it from learning
*site-correlated* shortcuts that generalise across many different
patients from the same centre — entirely compatible with a correct,
patient-grouped split, since that split was never designed to address
this axis. **Patient-level splitting is necessary but not sufficient;
site-level splitting (holding out whole sites, not just whole patients)
is the stricter standard this dataset would need to fully rule a site
confound out.**

**Checked before trusting the 84%, not assumed**: `build_split` groups
on `patient_id` only, not stratified on site — if sites happened to
cluster by fold, part of the 84% could be fold structure rather than
optics. Reproduced the split deterministically and tabulated every
site's count per fold: **all 20 classes appear in all three folds**,
train-fraction ranging 0.63–0.83 around the 0.70 target — reasonably
proportional, no site concentrated into one fold. The result stands.

**Then the direct experiment**: arm E holds out entire sites
(`splits.site_group_split`, same persisted-split discipline as the
item-1 fix). Patient-level integrity comes free — 99.0% of two-eye
patients share one site, so grouping by site overwhelmingly keeps a
patient's eyes together too — though not perfectly (6 patients straddle
a fold boundary under `site_group` vs 0 under `patient_group`, checked
directly). **A real caveat before the result**: with only 20 groups and
`site_0` alone holding 31% of the dataset, the val fold ended up being
a single site (402 images) and so did test (1982 images) — not one site
dominating a mixed fold, the fold *is* one site, and its class balance
(63% abnormal) differs substantially from train's (53%) as a direct
consequence of the entanglement just measured, not a split bug.

**Predicted before running anything: AUROC should drop substantially
relative to arm B.** It did, decisively — the cleanest result in this
project: AUROC **−0.0557** (over 3x the original patient-level effect),
5/5 seeds, survives Bonferroni correction with a CI excluding zero even
after correction (p=0.0026). Sens@95%Spec matches (−0.0585, p=0.0087,
survives). AUPRC alone doesn't move (+0.0023, p=0.706) — plausibly
*because of* the class-balance shift just flagged, not despite it:
AUPRC's precision baseline scales with test prevalence, while AUROC and
Sens@95%Spec are rank-based and prevalence-insulated by construction.
Predicted higher seed-to-seed variance too; found a suggestive (1.96x)
but not Levene-significant ratio for AUROC (p=0.451), no support at all
for Sens@95%Spec (0.86x, the *opposite* direction) — reported as
exactly that, not rounded up. Full per-metric table, confusion matrix,
per-fold site/class-balance tables, and the per-category chi-square
breakdown: `docs/notes.md`.

**The hierarchy this project's central claim rests on is now measured
at every rung, not just diagnosed at the top one.** Patient-level
splitting is necessary but not sufficient; this arm is the direct
demonstration of that, not just the statistical association behind it.

**A process note worth including on its own terms**: the first attempt
at the domain-shift classifier crashed *after* a 14-minute training run
completed, on a downstream bug (the canonical manifest doesn't carry
the original N/D/G/... columns; only `full_df.csv` does) that had
nothing to do with training at all. The fix wasn't just correcting the
bug — it was adding a checkpoint write immediately after the expensive
part and before the cheap analysis that crashed, so the next bug in
that cheap part (there wasn't one, but there could have been) would
never again cost re-running the 14 minutes to find out. Designing
around where a failure is expensive, not just fixing the failure
itself, is worth a sentence here.

**Arm E's headline number carried two open confounds when first
reported: its test set's class balance differs from arm B's (63% vs 55%
abnormal), and its test fold *is* a single site — one observation, not a
distribution. Both were checked directly rather than left as caveats.**

*The 6 straddling patients, first.* Verified the guess directly: 10
patients (not just the 6 that crossed a fold boundary — 4 more have
inconsistent labels that landed in the same multi-site train fold) really
do have two eyes at different raw resolutions, because resolution is a
per-*image*, not per-patient, property. Fixed with
`enforce_patient_site_consistency` — each patient's pair is forced to one
label (their majority, ties broken away from the `Other` bucket): 10
images reassigned, 0 patients left inconsistent, patient overlap on the
re-split falls to 0/0/0. Effect on the split itself is small, as expected
from a 10-image change on 6392: train 4008→4006, val 402→404, **and the
test fold — arm E's headline result — is untouched: same site (site_0),
same 1982 images.**

*Prevalence-matched re-evaluation* — retrained B and E (5 seeds,
`save_predictions=True` this time, since the original runs never
persisted per-example scores or weights, only aggregate metrics; closed
that gap in `train.run_train` to make this check possible at all).
Compared the AUROC gap at B's own 55% prevalence, at E's own 63%, and at
each subsampled to match the other. All three land in a **0.0032-wide
band** (−0.0514 to −0.0546) — under 6% of the gap's own size. **The drop
survives prevalence-matching in both directions, essentially unchanged.
It is not a prevalence artifact.**

*Leave-one-site-out*, across the 4 next-largest named sites (`Other`
excluded — it's a merged bucket, not a real site), 3 seeds each,
compared to the same 3-seed B baseline: 4 of 5 held-out sites (including
the original site_0) show the same negative gap, −0.033 to −0.053 —
this is not just the one site that happened to land in the original
split. **site_4 is a genuine exception: +0.0208, consistent across all
three of its own seeds**, so not a training fluke — and its class
balance (45.2% abnormal) doesn't explain it either, being unremarkable
next to two sites (site_1, site_3) that *do* show the expected drop.
Reported as found, not smoothed over: with only 5 site-level
observations this doesn't clear a two-sided significance threshold on
its own (t-test p=0.096, sign test p=0.375) — the honest read is
**"generalizes across most held-out sites, not all of them,"** which is
exactly the outcome this check was designed to be able to report either
way. Full numbers, per-seed and per-site: `docs/notes.md`.

**Two more checks on site_4, both humbling in useful ways.** First: a
confidence interval on each site's gap that accounts for *test-fold
size*, not just seed variance (closed-form Hanley-McNeil SE, using only
each fold's fixed positive/negative counts — no retraining needed).
This answers a different question than the 5-seed Bonferroni result
does — not "does this gap reproduce across models on this same fixed
fold" (yes, unaffected) but "would it survive a different, equally-sized
sample from that site's population." **Every single site's interval
spans zero, including the original site_0 result** — with one fold's
worth of images (336–1982), no individual site's AUROC is precise
enough alone to rule out a true gap of zero, symmetrically across all
five, not just the exception. The honest read: this weakens confidence
in any *one* site's number in isolation; it's the reason the project
runs a whole sweep rather than trusting a single held-out fold.

Second: does distance from the training distribution (mean pairwise
cosine distance, pretrained ResNet18 features — the exact extraction
`dedupe.py` already uses) predict the accuracy loss? Predicted site_4
would be *closest* to training, explaining its exception. **It isn't** —
it's tied for farthest, and site_0 (the largest loss) is the
*second-closest*, backwards from the hypothesis. Correlation is weak
and wrong-signed (Pearson r=+0.40, p=0.50) — reported as a clean
negative result, not reframed to look better. site_4 stays an
unexplained exception, which is a fine place to leave it.

## 10. Limitations

Stated plainly, because an interviewer will find these anyway and finding
them first is the better position:

- **~~Single dataset.~~ Now externally validated — with limits.** This
  read "everything here is ODIR-5K, no cross-dataset generalisation
  claim" until the audit was re-run end to end on EyePACS (35,126
  images, 17,563 patients, a US telemedicine screening network). The
  structural findings replicate: patient-level leakage 45.9% vs 46.5%
  like-for-like, bilateral correlation +25.5 vs +27.3 points over
  chance, and site recoverability *stronger* than ODIR-5K's (93.2% vs
  84.0% against near-identical baselines, χ²=139.6 vs 115.2). What does
  **not** transfer is every tuned threshold — see §11. Still only two
  datasets, both diabetic-retinopathy-weighted, and EyePACS's site
  labels are the same resolution *proxy*, not ground-truth camera
  metadata, so "site" remains a lower bound on distinct sources in both.
- **Binary task.** Normal/abnormal collapses 8 original categories.
  Assessed (not measured) to likely *understate* true bilateral-leakage
  sensitivity relative to a per-disease task — see §5.
- **Patient-level labels for the `normal_column` comparison arm.** Kept
  deliberately as a documented worse alternative, not removed, because the
  investigation that rejected it in favour of `keywords` is itself part of
  the project's evidence, not just its conclusion.
- **n=5 seeds is underpowered for precise effect-size claims**, openly
  quantified rather than glossed: the Bonferroni-adjusted confidence
  intervals for AUROC/AUPRC include zero, and detecting the observed
  Sens@95%Spec effect at 80% power would need ~31 seeds. Confidence
  intervals are reported specifically because they say what a bare p-value
  can't — the range of true effects the data are actually consistent with.
- **Quality and dedupe thresholds are evidence-based on this project's own
  investigation, not externally validated against ground-truth gradability
  or duplication labels** (none exist for this dataset). The uniform-haze
  blind spot in quality scoring (§6) is a concrete, known instance of this
  limitation, not a hypothetical one.
- **Site-level splitting (arm E) exists but isn't the default.** The
  cross-camera audit confirms a site confound (chi2=115.18, p=8.8e-16
  against diagnosis) and arm E measures its cost directly (AUROC drops
  0.0557 when entire sites are held out, Bonferroni-significant) — the
  main experiment matrix (arms A-D) still splits by patient, not site.
  The single-test-fold and prevalence-shift confounds this originally
  carried were followed up directly (§9): prevalence-matching shows the
  drop isn't a class-balance artifact, and leave-one-site-out across 4
  more sites replicates the direction for 3 of them (one, site_4, is a
  genuine unexplained exception not explained by class balance or by
  distance from the training distribution in embedding space) — so
  this is a real, largely site-general effect, not a single-fold fluke.
  A test-fold-size-aware confidence interval on *any individual* site's
  gap (including the original site_0) spans zero, though — 5 one-fold
  observations, however consistent in direction, is too few and each
  too small on its own to call this a precise, universal per-site
  estimate of a production model's site-level generalisation gap.

---

## 11. External validation on a second dataset (EyePACS)

**Why this was worth the compute.** Every finding above rests on ODIR-5K,
and the site result has an obvious deflationary reading: ODIR-5K
aggregates several Chinese centres with mixed camera stock, so a
recoverable "site" signal could be an artifact of *that aggregation*
rather than a property of fundus imaging. If so, the project's central
claim — patient-level splitting is necessary but insufficient — would be
parochial. The only way to settle it is a second dataset with a
different population, different equipment and a different collection
protocol.

**Choosing it (and what the survey ruled out).** EyePACS, APTOS 2019,
IDRiD, RFMiD, BRSET and Messidor were checked *before* downloading
anything, because the full EyePACS competition is ~82GB. Only EyePACS
combines per-image patient IDs, enough scale, and immediate
availability. APTOS and RFMiD publish image IDs only — which kills the
leakage half of the comparison. That one is worth a sentence at
interview: a web summary confidently stated RFMiD ships patient and eye
columns, and downloading its actual 183KB label CSV showed 46 disease
columns and no patient field at all. Checking the primary artifact cost
one minute and prevented a wasted 8GB download and a wrong conclusion.
IDRiD is single-camera, single-clinic (516 images — no site
heterogeneity to find *by construction*). BRSET has ideal structure but
needs PhysioNet credentialing. Only the labelled train split was pulled:
35,126 images, ~32.6GB instead of ~82GB, because every check needs one
labelled pool with patient IDs, not a train/test comparison.

**Did the adapter abstraction hold?** Largely, and the exceptions are
the interesting part. `ingest.py` needed no changes — it dispatches on
`cfg["dataset"]["name"]` and `adapters/__init__.py` auto-discovers
adapter modules via `pkgutil`, so a new dataset genuinely is one new
file. `quality.py`, `dedupe.py`, `splits.py`, `train.py` and
`experiment.py` contain no dataset-specific logic and were untouched.
One shared helper moved: `odir5k.py`'s private `_subsample_by_patient`
was needed verbatim, so it became `subsample_by_patient` in
`adapters/base.py`. That is a DRY fix a second adapter makes visible,
not a failure of the manifest contract.

### The result

| | ODIR-5K | EyePACS |
|---|---|---|
| Patients straddling `image_random` (two-eye only) | 46.5% | 45.9% |
| Fellow-eye concordance, lift over chance | +27.3 pts | +25.5 pts |
| **Site-classifier accuracy** | **0.8397** | **0.9317** |
| Majority-class baseline | 0.2932 | 0.2958 |
| **Site vs diagnosis** | χ²=115.2 | **χ²=139.6** |

**Site is more recoverable in EyePACS than in ODIR-5K**, on a test set
5.5x larger, from a single US screening network rather than a
multi-centre aggregation. The deflationary reading is dead.

Two comparisons would have been wrong if taken at face value, and both
are good interview material:

- **Leakage** looks like 42.0% vs 45.9%. But ODIR-5K has 324 single-eye
  patients who *cannot* straddle a split by construction; EyePACS has
  none. Restricted to two-eye patients the numbers are 46.5% vs 45.9% —
  the "difference" was a denominator artifact.
- **Concordance** looks like 77.8% vs 94.0%, a 16-point gap. But
  EyePACS is 80/20 normal against ODIR-5K's 55/45, so its chance floor
  is 68.5% rather than 50.5%. Lift over chance is +27.3 vs +25.5. The
  bilateral signal is nearly identical; the raw numbers were measuring
  class balance.

### What did not transfer, and the mistake I nearly published

Quality thresholds don't transfer — but the first version of that
finding was badly misleading, and catching it is the part worth
explaining. Naively EyePACS rejects 16.9% of images against ODIR-5K's
0.7%: a 24x gap that reads as "EyePACS is far worse quality." The
control that killed it: push ODIR-5K's *own raw images* through the
identical LANCZOS→512×512 path EyePACS went through, and its reject rate
moves 0.7% → 7.4%, median gradability 0.879 → 0.740, with no change
whatsoever to the photographs. Fair comparison is 2.3x, not 24x.

Roughly half the apparent gap was *preprocessing history*, not image
quality. This is `variance_of_laplacian`'s own documented failure mode —
"it ranks cameras, not sharpness" — entering one level earlier than the
module can defend against, since its internal resize cannot undo a
resize that already happened. The metric transfers; the absolute 0.5
cutoff is calibrated to a pipeline.

A related pleasure: 99.6% of EyePACS images trip `fov_clipped` (vs 6.9%
of ODIR-5K's — they're truncated ovals). `quality.py` deliberately does
*not* gate rejection on that flag, a decision made on ODIR-5K evidence
alone after circularity was found to conflate oval crops with real
truncation. Had it gated, essentially all of EyePACS would have been
rejected. An earlier judgement call turned out to be load-bearing for a
dataset it was never tested against.

### Duplicates: the largest divergence

At matched size — 6,392 images each, which is ODIR-5K's *whole dataset*
but a *subsample* of EyePACS's 35,126 — EyePACS has **444 verified
duplicate pairs against ODIR-5K's 11** (69.5 vs 1.7 per 1,000 images),
and **441 cases of the same eye filed under two different patient IDs**
against ODIR-5K's 8 (69.0 vs 1.3 per 1,000). The EyePACS figure is not a
dataset total: the full-dataset count is higher but unmeasured, and does
not scale linearly, since candidate pairs grow quadratically with n.

The obvious confound was tested before believing it: EyePACS has many
near-black failed captures, and two blank frames would pass both the
phash and the pixel-difference check while being unrelated photographs.
Duplicates *are* enriched for dark, low-quality images (median intensity
50.6 vs 73.3) — but not explained by them (median duplicate score 0.569,
minimum 0.271, not 0.00). Settled by eye, as ODIR-5K's phash
false-positive investigation was: a sampled contact sheet shows
unmistakably identical photographs — matching vessel trees, optic disc
positions, frame-edge notches — several differing only in white balance.

**The sharpest consequence: 68 of 118 duplicate clusters (58%) straddle
a `patient_group` split, and patient-grouping cannot prevent it.** The
two copies are *declared to be different patients*, so grouping on
`patient_id` is structurally incapable of keeping them together. §7
already argued dedupe closes a gap splitting cannot; EyePACS
demonstrates it at 34x the cluster count in a real screening archive.

### A scaling limit found the hard way

`phash_duplicates` builds a full n×n distance matrix via `squareform`:
0.46GB at ODIR-5K's n=6,392, but **13.79GB at EyePACS's n=35,126**
against 7.8GB of RAM. An O(n²) memory bug invisible at the scale it was
written against. Deliberately *not* rewritten: candidate counts scale
with n² as well (31,084 candidates at n=6,392 implies ~930,000 at
n=35,126, each needing two image loads to verify), so fixing the memory
would only expose a worse wall downstream. The honest answer was to run
the comparison at matched size and report the scaling limit as a
finding.

### Process failures worth owning

Two, both mine, both instructive.

First, I set `num_workers=6` to speed up data loading after measuring
that the GPU sat idle ~90% of the time. It trained all 10 epochs and
then died building the test loader: `WinError 1455, the paging file is
too small`. This machine has 7.8GB of RAM and each spawned worker
imports torch and reserves ~1.4GB for CUDA DLLs. The optimisation was
made without checking the memory budget, and it orphaned 17 worker
processes holding ~24GB of commit between them.

Second, and worse: that run lost ~70 minutes of *correct* training
because `best_state` existed only in RAM. §9's process note already
records this exact lesson from the manifest-columns crash — checkpoint
before the cheap step that can still fail — but it had never been
applied to the expensive step itself. Best weights now go to disk the
moment they improve, and the script resumes from them; verified by
reloading the 42.7MB checkpoint into a fresh ResNet18 rather than
assuming it worked.

---

## 12. Transitive chaining: a dedupe defect that only appears at scale

This one is worth reading even if you never touch fundus data. It is a
defect in the *method*, not in a threshold or a dataset, it is invisible
at small n, and anyone who builds duplicate detection by thresholding
pairwise distances and then grouping the survivors will eventually hit
it.

### The mechanism

`dedupe.py` finds duplicate *pairs* by thresholding a distance, then
groups them into clusters with union-find. Union-find implements
**single-linkage**: if A~B is a link and B~C is a link, A, B and C become
one cluster — regardless of how far apart A and C actually are. Nothing
ever checks A against C.

For a threshold to mean anything, a cluster should be a set of images
that are all duplicates of each other. Single-linkage does not produce
that. It produces sets connected by *some* path of short hops, which is a
much weaker property, and the difference between the two grows with the
density of links.

### The numbers

The same code, the same threshold, the same dataset — only n changes:

| | n=6,392 (subsample) | n=35,126 (full) |
|---|---|---|
| Largest cluster | 46 images | **1,870 images** |
| Second largest | — | **1,797 images** |
| Clusters ≥100 images | 0 | 5, holding **75% of all flagged images** |
| Median cluster size | 2 | 2 |

The median never moved. Most clusters are still honest pairs. But five
clusters swallowed three-quarters of the flagged images.

Randomly sampled members *within* those giant clusters measure **7.9–9.6
apart** — above the 5.0 threshold that supposedly defines membership.
They were never compared to each other; they were linked through
intermediaries. Sampled visually, they are plainly different eyes that
share a tone and a framing.

So at n=35,126 the word "cluster" stopped meaning what the threshold
defines, and the counts built on it — **16,782 pairs, 5,721 images, 492
clusters** — are inflated and are not used anywhere in this project.

### Why the subsample hid it completely

Chains need density. A chain A~B~C~D requires every consecutive link to
exist, and link count grows with the *square* of dataset size while image
count grows linearly: 31,084 candidate pairs at n=6,392 became 1,005,485
at n=35,126 (n^2.03). Each image therefore has ~5.5x more neighbours
within threshold in the full dataset, which is exactly the condition
under which long chains stop being rare and start being inevitable.

This is the uncomfortable part: **the subsample was not a smaller version
of the same result, it was a qualitatively different regime.** Validating
a clustering method at one scale says very little about its behaviour at
another, and nothing at all about a failure mode whose trigger *is*
scale.

### A smaller threshold is not the fix

The obvious reaction — tighten the threshold until the giant clusters
break up — treats a symptom. Chaining is a property of single-linkage,
not of where the cutoff sits: a tighter threshold thins the link graph so
chains need more data to form, but the failure returns at larger n, now
harder to notice because the clusters look reasonable for longer. On this
data it would also be actively destructive, since EyePACS has no pairs
below 2.0 at all, so tightening toward ODIR-5K's near-zero duplicates
discards the entire finding rather than cleaning it.

**What would actually work**, in increasing order of cost:

1. **Complete-linkage within each connected component.** Build components
   with union-find as now, then inside each one require that *every* pair
   is within threshold, splitting until that holds.
2. **A diameter constraint**, rejecting or flagging any component whose
   maximum internal distance exceeds the threshold.
3. **Maximal-clique enumeration** on the threshold graph, which is the
   exact formulation of "everyone is a duplicate of everyone".

**I would implement (1).** It enforces precisely the property the
threshold is supposed to assert, which is the actual defect, and it is
affordable where it needs to be: connected components are cheap to build,
and the O(k²) all-pairs check is only paid inside each component — which
is negligible for the 2-image clusters that dominate, and concentrated
exactly on the pathological ones that deserve the scrutiny. Option (2)
diagnoses the problem without repairing it: it tells you a cluster is
untrustworthy but not what the honest sub-clusters are. Option (3) is the
theoretically exact answer but NP-hard in general, and buys little over
(1) once components are small.

Deliberately not implemented here. This project's duplicate claims are
reported at the **pair** level, and pairs are individually verified and
cannot chain — so the defect changes no number that is actually used. The
right time to build (1) is when cluster-level output is needed at scale,
and it should be validated at full n, because the subsample regime above
demonstrates that passing at small n proves nothing.

### Two corrections, a day apart, in opposite directions

Both errors came from reasoning about numbers instead of looking at
images, and they failed in mirror-image ways.

**Over-claiming.** Mid-run, with the job still going and only CPU-time to
go on, I inferred that candidates were growing as ~n^2.55 and offered a
mechanism: dense cliques of near-identical failed captures. The full scan
measured 1,005,485 candidates against a quadratic prediction of 938,687 —
**n^2.03, essentially exactly quadratic.** The runtime overrun was
per-pair verification cost, not candidate count. I proposed a mechanism
before running the measurement that would have tested it.

**Over-correcting.** A day later, on discovering the chaining above, I
declared the EyePACS duplicate finding a threshold artifact outright.
That was equally unfounded. The two pieces of evidence I leaned on
dissolve on inspection: the frame-edge "fingerprint" notch is a
systematic EyePACS capture artifact appearing across unrelated images
(so it proves nothing either way), and the fact that the
lowest-difference pair is a patient's own two eyes shows the metric is
*noisy*, not that every pair above it is spurious.

The pattern worth extracting: the first error trusted a number without an
image, and the second trusted a different number without an image. **Both
were resolved in minutes by rendering the actual photographs** — first
the giant clusters (different eyes, chaining confirmed), then
full-resolution difference maps (same eyes, finding restored).

### The technique that settled it: cross-dataset calibration

The difference maps were initially unreadable. Every EyePACS pair showed
bright vessel-shaped residuals, and I could not tell whether that meant
"different eyes" or "same eye, slightly re-registered" — the whole
question, and a scalar distance cannot answer it.

What resolved it was a **known-good case from the other dataset**.
ODIR-5K has a verified duplicate pair sitting near the threshold at 4.89,
independently confirmed genuine, and it is unambiguously the same eye.
Its difference map shows vessel-shaped residuals *identical in character*
to EyePACS's. That fixes the interpretation: on this modality, vessel
residual indicates sub-pixel registration shift, not different eyes.

Generalising the move: **when a diagnostic is ambiguous on the dataset
you are judging, calibrate it on a confirmed example from a dataset where
you already know the answer.** ODIR-5K's byte-identical duplicates and
its near-threshold duplicate together span the range and turn an
unreadable image into a readable one. It costs one lookup and it converts
an argument into an observation. This project needed two datasets for its
headline result anyway; that it also supplied a calibration standard was
an unplanned benefit of having one.

### What the distance distributions imply about provenance

One further observation, **flagged as an inference from the distance
distribution rather than something confirmed** — no provenance metadata
was consulted, and none is published with either dataset.

- **ODIR-5K**: verified pairs span 0.0037 to 4.89, and the tightest are
  pixel-identical — an empty difference map.
- **EyePACS**: across all 31,084 candidates, **not one pair falls below
  2.0**. The 444 verified pairs occupy 2.0–5.0 in a smooth continuum.

A pixel-identical pair is most simply explained by the same file being
stored twice. A population that never gets closer than 2.0, yet shows the
same eye with matching disc, arcade and macula, is most simply explained
by the same capture being re-exported or re-graded — a colour or
compression change applied once, which cannot produce a zero difference
however identical the underlying photograph.

If that reading is right, the two datasets are exhibiting **different
duplication mechanisms**: literal file duplication in the curated
research collection, versus re-processing and re-enrolment of the same
capture in the operational screening archive. That would fit how the two
were assembled, and it would mean a dedupe threshold calibrated on
file-level duplicates is structurally mismatched to an archive whose
duplicates are re-processed. Consistent with the evidence, not
established by it — confirming it needs provenance data neither dataset
ships.

---

## 13. A provenance finding: the multi-grader data these datasets are famous for

Before any segmentation code was written, Step 0 for item 4 checked what
is actually obtainable. Two of the most-cited segmentation datasets in
retinal imaging turn out not to ship, through their primary channels, the
multi-grader annotations their reputations rest on. This is a finding
about data provenance rather than about models, and it determines whether
an inter-grader ceiling can be computed at all.

### DRIVE's second observer: present in the literature, absent from the official distribution

DRIVE's test set is documented as carrying manual vessel delineations
from **two** independent human observers, and the second-observer
agreement is quoted throughout the vessel-segmentation literature as the
human performance ceiling. What the official channel actually
distributes is different. The DRIVE Grand Challenge download page states
plainly:

> "For the test cases no annotations are made available, you will be able
> to submit your predictions to this site and have them compared to the
> gold standard."

The test annotations — first and second observer both — are retained
server-side for scoring. Registration gets you the images, not the
labels.

**What was checked, and what each source contains.** 22 DRIVE-candidate
datasets on Kaggle were enumerated by API (file listings only, nothing
downloaded), plus a local copy already on disk:

| Source | `test/1st_manual` | `test/2nd_manual` |
|---|---|---|
| Official Grand Challenge distribution | withheld (server-side scoring) | withheld |
| `andrewmvd/...` — top result, 18.8k downloads, 0.875 usability | absent | absent |
| `srinjoybhuiya/...`, `namnguynnnn/...`, `zhz638/...`, `vasavigneswar/...` | absent | absent |
| `tushartalukder/...`, `yattseung/...`, `anacondaece/...`, `vutu123456/...` | absent | absent |
| `ipythonx/...` (470MB multi-dataset collection) | absent | absent |
| `pradosh123/...` "Test/Masks" | **not DRIVE** — files are HRF (`13_dr_HRF.tif`) | — |
| local copy (`test.zip` / `training.zip`) | absent | absent |
| **`ahtcmstp/retina`** | **present (20)** | **present (20)** |
| **`xxc025/111111`** | **present (20)** | **present (20)** |
| `zionfuo/drive2004`, `a1742976730/...` (inside `Retina-Unet-master/`) | — | present (20) |

**The claim, stated precisely.** The second-observer set is *not*
unobtainable — it exists and circulates. It is absent from the official
distribution and from every curated, highly-downloaded mirror checked,
and survives in a handful of obscure re-uploads, several of which are
clearly copies of the pre-Grand-Challenge distribution (one is named
`drive2004`; another is bundled inside a checkout of an old
`Retina-Unet` repository). This says what the sources checked contain.
It does not establish that no other public distribution has it, and it
makes no claim about what is available under institutional agreement.

**Why this matters for reproducibility.** Anyone following the
authoritative route — register at the official site, download DRIVE —
cannot compute the second-observer agreement that their own field quotes
as the ceiling. Anyone who happens to pull an old third-party re-upload
can. The number is reproducible only by accident of which copy you
obtained, and the copies are not distinguished by name, size, or
description. The most-downloaded, highest-usability-rated mirror is
among those missing it, so ordinary care in choosing a source selects
*against* the complete version.

An earlier draft of this section claimed the second-observer data was
simply unavailable, on the strength of five sources. Widening to 22
refuted that within minutes. Recorded here because the wrong version was
one edit away from being published, and the only thing that prevented it
was checking more sources before asserting a negative.

### REFUGE: seven graders, one released reference

The same shape of gap, for the same reason, in the other dataset item 4
considered. REFUGE's optic disc and cup annotations were produced by
**seven independent glaucoma specialists**, then merged by a senior
specialist into a single reference standard. The merged reference is what
ships; the seven individual annotations are not released.

So REFUGE — 1,200 images, the largest and most rigorously annotated of
the candidates — cannot supply an inter-grader number either, despite
having *more* grader redundancy behind it than any other dataset here.
The variance was measured and then averaged away before distribution.

### The consequence

Two of the most-cited retinal segmentation datasets, and neither ships
usable multi-grader data through its primary channel: DRIVE withholds it,
REFUGE merges it. **CHASE_DB1 carries this project's inter-grader work
by default rather than by preference** — 28 images, two observers
(`1stHO`/`2ndHO`), 56 masks, verified present by file listing.

There is a general lesson worth stating for anyone planning work that
depends on annotation variance: **confirm the multi-grader data is in the
distribution you can actually obtain, before designing around it.** A
dataset's reputation reflects what was collected, which is not the same
as what is published. Checking costs an API call; discovering it after
building costs the experiment.

---

## 14. The first thing that transferred

Every cross-dataset comparison in this project so far has come apart on
contact. Then one did not, and the contrast is the point.

### What did not transfer

| Quantity | ODIR-5K | EyePACS | |
|---|---|---|---|
| Quality reject rate at the 0.5 cutoff | 0.7% (as shipped) | 16.9% | threshold is calibrated to a *preprocessing pipeline*, not a dataset — re-deriving ODIR-5K through the same resize moved it to 7.4% |
| Duplicate pairs per 1,000 images | 1.7 | 69.5 | ~40x, at matched sample size |
| Dedupe pixel-difference threshold | separates cleanly (genuine pairs at 0.0037, false positives at 15+) | no separation at all — nothing below 2.0, smooth continuum through the cutoff | |
| Site recoverability | 84.0% | 93.2% | transferred in *direction*, not in value |

The pattern was consistent enough to become an expectation: numbers tuned
on one dataset describe that dataset, and porting them produces
confident nonsense. Two of this project's corrections came from exactly
that.

### What did transfer

Inter-grader agreement on vessel annotation:

| | n | Dice | 95% CI |
|---|---|---|---|
| CHASE_DB1 | 28 | **0.7765** ± 0.0250 | [0.7673, 0.7858] |
| DRIVE | 20 | **0.7879** ± 0.0206 | [0.7789, 0.7969] |

**0.0114 apart, with overlapping confidence intervals.** These are
different annotators, in different countries, working from different
cameras at different resolutions (999×960 versus 565×584), under
annotation efforts separated by roughly a decade and with no shared
protocol. CHASE_DB1's images are of schoolchildren in England;
DRIVE's are from a Dutch diabetic-retinopathy screening programme.

Two independent groups of humans, given the same kind of task, disagree
with each other by the same amount.

### Why that is interesting rather than a curiosity

The quantities that failed to transfer were all **properties of a
dataset or a pipeline**: how a collection was preprocessed, how often it
contains duplicates, how its cameras vary. It makes sense that those
differ, and in hindsight expecting otherwise was the error.

Inter-observer agreement is not that. It is a property of **the task** —
of how much genuine ambiguity there is in deciding where a vessel ends
and the background begins, at the resolution fundus photography offers.
That ambiguity lives in the anatomy and the imaging, not in the
institution, so a stable value across two unrelated annotation efforts
is the outcome the hypothesis predicts.

Stated carefully, because two datasets is two datasets: **this is
consistent with ~0.78 Dice being a property of vessel annotation itself
rather than of either annotation effort.** It is not established by n=2.
A third two-observer vessel dataset landing near 0.78 would make the case
considerably stronger, and one landing at 0.85 would refute it outright.
The prediction is falsifiable, which is the main thing to like about it.

### The practical consequence

If ~0.78 is the task's ceiling rather than a quirk of one dataset, then
it is the number any vessel-segmentation result should be read against,
regardless of which dataset produced it. A model reported at 0.80 Dice
is not 20 points from perfect. It is at or slightly past the point where
two human experts stop agreeing with each other — which raises a
question worth taking seriously, addressed in §15.

---

## 15. What does a Dice above inter-observer agreement mean?

A recurring figure in the vessel-segmentation literature is **0.80-0.82
Dice** against DRIVE's first observer. Both ceilings measured here sit
below that: 0.7765 on CHASE_DB1, 0.7879 on DRIVE.

**What is not being claimed.** No paper has been audited here. No
specific published result is being called wrong, and none could be on
this evidence — this project trained no vessel model at the time of
writing and re-ran nobody's code. The observation is that two numbers
which are usually quoted separately sit in a surprising order when put
next to each other.

**The benign explanations are real and probably sufficient.** At least
two are strong enough that they should be the default reading:

1. **Models are graded against observer 1 alone.** Observer 1's
   annotations are the reference standard; observer 2's exist only as a
   human comparator. A model trained on observer 1's conventions is
   optimising toward one annotator's specific habits — where they place
   a boundary on a faint capillary, how far down the vessel tree they
   keep labelling. Matching those habits more closely than a second
   human does is an entirely coherent thing for a fitted model to
   achieve, and it is not evidence of superhuman vessel perception. It
   is evidence of successfully fitting a style.
2. **These ceilings may not describe the data a given paper used.** They
   were measured here, on these copies, with this binarisation. A paper
   using a different preprocessing pipeline, a different test split, or
   a differently-sourced copy of DRIVE is not necessarily operating
   against the same ceiling — this project has already been caught out
   once assuming a threshold transfers across preprocessing (§11), and
   the same caution applies to itself.

**The question worth asking.** Given that the ground truth is one
observer's opinion, and that a second qualified observer reproduces it
only ~78% by Dice, what does a score above ~78% actually certify? Three
readings, and the field is better placed than this project to say which
holds:

- the model genuinely delineates vessels better than the second observer,
  and observer 2 is simply the weaker annotator;
- the model has fitted observer 1's idiosyncrasies, and the excess above
  the ceiling measures style-matching rather than accuracy;
- the metric saturates in a way that makes small differences near the
  ceiling uninformative, so 0.80 and 0.78 are not meaningfully different
  measurements at all.

These have different consequences. Under the first, the number means what
it appears to. Under the second, leaderboard gains above the ceiling
partly measure conformity to one person, and a model tuned that way may
transfer poorly to a site whose graders annotate differently — which
would connect directly to the site-generalisation finding in §9-§11.
Under the third, a good deal of reported progress is inside the noise.

**A concrete way to settle it**, cheaper than re-auditing the
literature: score a model against observer 2 as well as observer 1. If
performance drops to around the inter-observer level, the model has
learned observer 1's style. If it holds, it has learned vessels. That
comparison requires only the second-observer masks, which — as §13
documents — are the very thing the official distribution withholds. The
**A concrete way to settle it**, cheaper than re-auditing the
literature: score a model against observer 2 as well as observer 1. If
performance drops to around the inter-observer level, the model has
learned observer 1's style. If it holds, it has learned vessels. That
comparison requires only the second-observer masks, which — as §13
documents — are the very thing the official distribution withholds. The
data needed to check the question is the data that is hardest to obtain,
which may be part of why the question is not routinely asked.

That check was then run on a model trained for this project. §16 reports
it, including the fact that the prediction above did not survive it.

---


## 16. The commensurable test: one model, both observers

§15 ends by proposing a check. This section is that check, run.

The optic disc model could not be used for it — a disc is one convex
blob where boundary disagreement is a small share of area, so disc Dice
normally runs 0.90+, while vessels are almost entirely boundary. Putting
a disc score beside a vessel ceiling would be a category error. So a
U-Net was trained on DRIVE's 20 training images and evaluated on DRIVE's
20 test images — the same images §14's ceiling was measured on, which is
what makes the comparison commensurable rather than indicative.

| | Dice | 95% CI |
|---|---|---|
| model vs **observer 1** (its training target) | 0.7884 ± 0.0229 | [0.7784, 0.7985] |
| model vs **observer 2** (never trained on) | **0.8074** ± 0.0258 | [0.7961, 0.8187] |
| observer 1 vs observer 2 (the ceiling) | 0.7882 ± 0.0208 | [0.7791, 0.7973] |

### The prediction failed, and it is kept here rather than quietly fixed

§15 predicted that a model graded against observer 1 would fit that
annotator's conventions, and would therefore score *better* against
observer 1 than against observer 2. **The sign is reversed.** The model
agrees more with the observer it never saw. The style gap is −0.0190
where the hypothesis required a positive value.

What the numbers do support is narrower and more interesting: the
model's agreement with observer 1 (0.7884) is statistically
indistinguishable from observer 2's agreement with observer 1 (0.7882).
**It reached inter-observer agreement and stopped** — exactly as close to
the reference standard as another qualified human, no closer and no
further.

One trap recorded because it looked like a result: the model "beat the
ceiling" on 13 of 20 images. That sounds striking and means almost
nothing, since the two means differ by 0.0002 — which of them is higher
on any given image is close to a coin flip. A per-image win rate can
sound like evidence while the underlying distributions are identical.

### Where the disagreements sit

Aggregate Dice cannot distinguish three quite different situations, so
the disagreement *sets* were compared directly rather than their sizes.
D_human = XOR(obs1, obs2), the pixels the humans dispute.
D_model = XOR(model, obs2), where the model departs from observer 2.

**The null matters and is stated explicitly.** Disagreements can only
occur where at least one annotator marked something, so the containing
region is U = obs1 ∪ obs2 ∪ model. If D_model were scattered at random
within U, the expected share landing inside D_human is |D_human| / |U|.
Observed over expected is an enrichment factor, where 1.0 means no
better than chance.

| | |
|---|---|
| observed concentration | 0.4987 |
| null expectation | 0.3199 |
| **enrichment** | **1.57x**, 95% CI [1.50, 1.63] |
| side-taking (fraction backing observer 1) | **0.4452**, 95% CI [0.4280, 0.4624] |

All three candidate explanations get a partial answer, and none wins
outright:

1. **It learned the easy consensus — partly true.** Errors are enriched
   1.57x in already-contested territory. The model left contested pixels
   contested.
2. **It makes its own distinct mistakes — also partly true.** Only half
   its errors are in contested territory; the other half sit elsewhere.
3. **It takes observer 1's side — refuted, and on the wrong side of
   chance.** Within contested pixels the model backs observer 1 only
   44.5% of the time, a confidence interval that excludes 0.5 in the
   direction opposite to the hypothesis. It backs observer 2 — the
   annotator it never trained on — 55.5% of the time.

An implementation note worth stating because it saves a measurement:
side-taking follows from binarity. Where obs1 ≠ obs2 (contested) and
model ≠ obs2, then model = obs1 necessarily. So
|D_model ∩ D_human| / |D_human| *is* the fraction siding with observer 1;
it does not need to be computed separately.

### The mechanism, and a prediction that would falsify it

Foreground area explains the direction the style-fitting hypothesis got
wrong:

| annotator | mean foreground area |
|---|---|
| observer 1 (training target) | 8.76% |
| **model** | **8.53%** |
| observer 2 | 8.44% |

The model landed *between* the two humans and within 1% of observer 2,
without ever seeing observer 2's masks. Observer 1 is the more liberal
annotator — marking faint capillaries further down the vessel tree —
and a Dice+BCE objective rewards confident, well-supported foreground.
That pulls the model toward conservative predictions and therefore
*away* from its own training annotator, which is precisely the observed
effect.

**This makes a testable prediction, stated before anyone runs it.** If
the conservatism is the objective's doing rather than something about
observer 2, then re-training with a recall-weighted loss — Tversky with
β > α, or weighted BCE favouring false positives over false negatives —
should push predicted foreground area above 8.76% and **flip side-taking
above 0.5**, because a liberal model would start backing the liberal
annotator. If side-taking stays below 0.5 under a recall-weighted loss,
the explanation offered here is wrong and something else is producing
the alignment with observer 2.

Not run, and deliberately flagged as untested rather than implied. It is
cheap — one retraining run on 16 images — and it is the obvious next
experiment for anyone continuing this work.

---

## 17. Three failures, and the curation story that wasn't true

The optic disc model reports Dice 0.8581 ± 0.1646 on IDRiD's 27 test
images. The mean is the wrong summary and the median (0.9110) is the
right one: **three images fail badly, two of them below 0.50 Dice.**
Three images out of 27 move the headline by 0.05, which is exactly the
underpowered regime declared before the run.

A 7% silent-failure rate matters more for a screening tool than a mean
does, so the three were investigated rather than averaged away.

### The convenient story, and why it is false

The satisfying result would have been that these are bad photographs and
that this project's own quality module would have rejected them upstream
— a clean connection between the curation work and the segmentation
work. The data refuses it:

| | failures (n=3) | rest (n=24) |
|---|---|---|
| gradability score | **0.776** | 0.708 |
| flagged at the configured 0.5 threshold | **0 of 3** | 0 of 24 |

**The failures are cleaner than average.** The quality module flags none
of them, and would not have caught a single one. This is a model
limitation, not a curation gap, and it is reported as one.

### What actually goes wrong

Group means are the wrong summary at n=3 — they average away the fact
that the three fail in two different ways — so the three are listed
individually, with the other 24 as a reference row:

| image | Dice | pred/true area | centroid err | components | exudate rank |
|---|---|---|---|---|---|
| IDRiD_62 | 0.356 | 4.62x | 25.2 px | 4 | **18** |
| IDRiD_66 | 0.420 | 3.22x | 130.5 px | 5 | **1** |
| IDRiD_55 | 0.554 | 0.46x | 31.3 px | 2 | **2** |
| *other 24* | *0.902 median* | *1.00x* | *5.4 px* | *1 median* | |

Two over-segment into four and five disconnected blobs; the third
collapses to under half the true disc area. All three put the disc centre
somewhere it is not. Rendered alongside their images, the cause is
visible in two of them: the model is latching onto **hard exudates** —
bright yellow-white lesions that look very much like an optic disc — in
a dataset whose entire purpose is diabetic retinopathy.

The exudate masks IDRiD ships allow this to be checked rather than
eyeballed, and the check is a strong one: exudate fraction 0.0398 across
the failures against 0.0072 for the rest, a **5.51x** ratio, and **the
most and second-most exudate-heavy images in the entire test set are both
failures**.

What the table makes visible, and a group mean would have hidden, is that
the fit is not uniform. IDRiD_66 matches the mechanism on both counts —
rank 1 for exudate, and five bright blobs 3.2x too large. IDRiD_55
matches on burden (rank 2) but *under*-segments, which is the opposite
geometric signature. So "exudate confusion" is carried by the burden
association for one of them and by burden plus geometry for the other,
and that distinction is worth keeping rather than smoothing into a single
2.77x mean.

### The clinical implication, stated plainly

Disc localisation degrades as exudate burden rises. **The failure mode
gets worse exactly on the sickest eyes** — the ones a screening tool
exists to find. That is backwards, and an ophthalmologist would
recognise it immediately: a disc segmenter that quietly breaks on florid
exudative retinopathy is unreliable precisely where reliability matters,
and its aggregate Dice will not show it, because sick eyes are a
minority of any test set.

It also has a concrete consequence for anything built downstream. Disc
segmentation is usually a *preprocessing* step — for cup-to-disc ratio,
for vessel-origin registration, for centring crops. A step that fails
silently on the most diseased images propagates that failure into
everything after it, and the symptom will appear somewhere else entirely.

### What is not explained

Two of three, not three of three. **IDRiD_62 ranks 18th of 27 by exudate
burden** and fails anyway, with a displaced and oversized prediction on a
relatively clean image. The exudate mechanism does not account for it and
no alternative is offered here.

The statistics deserve the same restraint: Mann-Whitney on exudate
burden between the two groups gives p=0.0595 at n=3 versus 24. That is
indicative and not significant, and the visual evidence plus the
rank-1-and-2 placement is doing more work than the test is. With three
failures, no stronger claim is available.

---

## Interview questions

1. **Why does patient-level splitting matter more here than in, say, chest
   X-ray classification?**
   Because retinal imaging has a mechanical two-samples-per-subject
   structure (left/right eye) plus real bilateral disease correlation, so
   a naive split doesn't just risk leaking *a* correlated sample — it
   leaks one specifically 42.0% of the time by construction (measured on
   this dataset), and that leaked sample shares the true label 77.8% of
   the time (vs. 50.5% by chance). Many chest X-ray datasets have one film
   per patient per encounter, so the *image*-vs-*patient* distinction that
   drives this whole project collapses for them — their leakage risk comes
   from other sources (repeat visits, near-duplicate captures), not from
   this specific structural doubling.

2. **Your gradability score is hand-weighted. Why not learn it?**
   Two reasons. First, there's no ground-truth gradability label for this
   dataset to learn *against* — any learned score would be trained on a
   proxy (e.g. downstream model performance, or a small hand-labelled
   subset) that introduces its own validation problem, one level removed.
   Second, "why was this image rejected" needs to be a one-sentence,
   inspectable answer for this project's purpose (auditing a pipeline,
   not maximising a metric) — a transparent weighted sum gives that for
   free; a learned classifier is a second black box on top of the model
   this project is already trying to keep honest about.

3. **How would you validate the quality thresholds without ground-truth
   quality labels?**
   Exactly what this project did: score everything, then look at the
   distribution's extremes by eye (a contact sheet of the lowest- and
   highest-scoring images) and check the direction is sane; inspect every
   actual reject individually rather than trusting the aggregate reject
   rate; and get an independent comparison population — scoring the raw,
   not-yet-curated `Training Images/` with the identical code and
   thresholds gave 4.2% vs. 0.7% on the curated set, a real, externally-
   anchored contrast rather than a number checked only against itself.

4. **What breaks if the same patient appears under two different IDs?**
   Patient-grouped splitting breaks silently — it groups by *declared*
   patient ID, so two different IDs that are actually the same person's
   capture are treated as unrelated, and one copy can land in train while
   the other lands in test with zero warning. This project found exactly
   this: 8 genuine cross-patient duplicate pairs (pixel-verified), which
   `patient_group_split` has no way to catch — only content-based
   deduplication does. It's the concrete argument in this codebase for why
   the pipeline needs both a grouped split *and* a dedupe stage, not just
   one.

5. **How would this pipeline change for a real hospital PACS instead of a
   Kaggle zip?**
   The adapter layer is exactly the seam designed for this — a new adapter
   function producing the same canonical manifest, nothing else in
   ingest/split/quality/dedupe/train changes. In practice: patient IDs
   would come from an MRN or similar (already assumed to be trustworthy
   here, which the duplicate-ID finding shows is not automatically safe
   even in a *curated* public release, let alone a live system without an
   equivalent audit); images would need DICOM handling and PHI stripping
   (explicitly called out as unbuilt in the README's roadmap); and the raw
   `Training Images/` reject-rate comparison in §6 is a preview of what a
   less-curated real intake stream would look like against these same
   thresholds.

6. **Horizontal flip augmentation on fundus images — safe or not, and
   why?**
   Not safe, and not used here. A flip turns a left eye into an
   anatomically wrong right eye — optic disc position relative to the
   macula is a real, learnable anatomical cue (nasal vs. temporal), and
   flipping destroys that relationship rather than teaching genuine
   rotational invariance. It's also leakage-adjacent: flipping could let a
   model exploit fellow-eye mirror symmetry as a shortcut, which isn't the
   kind of generalisation the eventual metric should be crediting.

7. **Your labels are patient-level but your rows are eye-level. What does
   that do to the metrics?**
   This is exactly the question the label-strategy investigation in §2
   answered empirically rather than by assumption: under `normal_column`
   (patient-level), a unilaterally-diseased patient's healthy fellow eye
   gets mislabelled abnormal, every time — confirmed by eyeballing 10
   sampled disagreements, all the same failure mode. That's a real,
   measurable label-noise cost (12.1% of rows disagree between the two
   candidate label sources), not a hypothetical one, and it's why the
   project default switched to the per-eye `keywords` label instead.

8. **How would you detect a camera or site confound?**
   Not built here (see Limitations), but the method already used
   elsewhere in this project generalises directly: stratify the quality
   score distribution or the model's error rate by whatever proxy for
   camera/site is available (image resolution before preprocessing,
   metadata fields if present) and check whether either shifts
   meaningfully between groups, the same way class balance was checked
   before/after curation and reject rate was checked raw-vs-preprocessed.
   A resolution-based proxy is plausible here specifically because the raw
   `Training Images/` already showed 12+ distinct resolutions, which is
   circumstantial evidence of a real camera mix worth stratifying by.

9. **What would you do differently if the downstream task were
   segmentation rather than classification?**
   The leakage argument is unchanged (a segmentation model can memorise a
   leaked fellow-eye's mask-relevant structure exactly the way a
   classifier memorises its label), so the split/dedupe machinery carries
   over directly. What changes: the metrics (Dice/IoU instead of
   AUROC/AUPRC/Sens@Spec), the quality gate (a segmentation task cares
   about boundary sharpness in specific anatomical regions more than
   global gradability), and critically, the label pipeline — this project
   sidestepped by using an existing per-eye label; a segmentation project
   would need mask-level QC (empty masks, area outliers, annotator
   agreement) as its own curation stage, which is exactly the kind of
   thing flagged as future work in the README roadmap (REFUGE/IDRiD
   segmentation, inter-grader agreement via DRIVE).

10. **Sensitivity at 95% specificity — why that operating point?**
    Because a screening tool's real deployment constraint is usually
    expressed as an acceptable false-positive (unnecessary-referral) rate,
    not an abstract threshold-independent ranking quality — 95% specificity
    is a conventional, clinically legible choice for that constraint in
    ophthalmic screening contexts. Reporting sensitivity *at* that fixed
    point, rather than AUROC alone, answers the question a clinician
    actually has: "at the false-positive rate my clinic can absorb, how
    many real cases does this model miss?"

11. **If the A-vs-B gap had come out near zero, what would you conclude?**
    This isn't hypothetical — it's close to what actually happened on the
    second full run (after arms C/D existed and forced a slightly
    different, more tightly matched training-set cap): the gap dropped
    from +0.0173 (significant uncorrected, 5/5 sign-consistent) to +0.0079
    (p=0.29, 3/5 sign-consistent). My conclusion in that situation: the
    *overlap* is still real and exact (42.0% of patients, unconditionally
    true regardless of any downstream model result), but this particular
    model/task/dataset/seed-count combination isn't reliably showing a
    detectable downstream effect — and I'd say so plainly rather than
    quietly reporting only the first, more favourable run. This project
    also practices the same stance on the curation arms: C and D were
    predicted in writing, before running them, to be statistically
    indistinguishable from B. That held for D. It did *not* fully hold for
    C, which showed a small but 5/5-consistent AUROC *decrease* — and I
    report that as the surprising result it is, with a hypothesis for why
    (curation matches training-set size, not composition), rather than
    rounding it back to the predicted null because the null was what I
    expected to find.

12. **What is the single weakest part of this project?**
    n=5 seeds — and this project has direct, empirical proof of it, not
    just a power calculation. Re-running the A-vs-B comparison a second
    time, under nearly identical conditions (same nominal seeds, training
    set matched within 0.85% of the original), was enough to flip the
    result from "significant, unanimous direction" to "not significant,
    3/5 agreeing." Every other number in this project — the 42.0%
    overlap, the 77.8% concordance, the 8 verified duplicate pairs, the
    0.7% reject rate — is either an exact count or independently
    cross-checked by eye. The one number that matters most for the
    headline claim is the one resting on the thinnest statistical
    foundation, and now has a failed replication attached to it as direct
    evidence of exactly that. More seeds is the honest fix, not a smaller
    correction, a friendlier test, or reporting only the run that came out
    cleaner.
