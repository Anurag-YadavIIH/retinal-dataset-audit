# Walkthrough

> Written for a reader who has to defend this project in a technical interview.
> **Read this file before reading the code.**

## 1. The claim

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
confound out.** Not implemented here — diagnosed and stated as a design
conclusion for future work, the same way the split-then-curate fix
above started as a diagnosis before it became a fix. Full numbers,
confusion matrix, and per-category chi-square table: `docs/notes.md`.

## 10. Limitations

Stated plainly, because an interviewer will find these anyway and finding
them first is the better position:

- **Single dataset.** Everything here is ODIR-5K. No external validation
  set, no cross-dataset generalisation claim. A leakage effect measured on
  one dataset's specific camera mix, label noise, and patient demographics
  may not transfer in magnitude to another.
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
- **No held-out camera/site split, and this is no longer a hypothetical
  gap.** The cross-camera audit above confirms a site confound exists
  (chi2=115.18, p=8.8e-16 against the patient-level diagnosis flag,
  significant for 6 of 8 individual categories) — patient-level
  splitting does not address it, since it groups by patient, not by
  site. Diagnosed and reported, not fixed: this project does not
  implement a site-level split.

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
