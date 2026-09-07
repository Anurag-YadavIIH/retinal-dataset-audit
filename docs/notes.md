# Working notes

Scratch space. Record decisions here as they are made so WALKTHROUGH.md can be
written from evidence rather than reconstructed from memory.

## Log

- [ ] Session 1: scaffold created, ingest + split + A/B experiment.

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
structurally necessary**: of the 8 confirmed pairs, phash alone would
have found all 8 (verified directly -- see above) but embeddings alone
would have found only 7, permanently blind to 3297/4542 the same way
phash alone was blind to nothing here (phash caught every pair; embedding
methods and phash fail in *different* directions on this modality, not
interchangeable ones -- see WALKTHROUGH.md 7 for the connection to the
phash false-positive finding, which is the same underlying pattern from
the opposite side).

**Final result**, phash-verified (8 pairs) + embedding at the corrected
0.99 threshold (10 pairs, 6 overlapping with phash): **11 distinct
duplicate clusters, 22 images, across 8 unique patient-pairs** (3 pairs
duplicated on both eyes, 5 on one eye). None are same-patient/fellow-eye;
all are cross-patient -- the same real capture filed under two different
declared patient IDs.

**Money metric**: of these 18 verified pairs (8+10, double-counting the 6
found by both methods -- the 11-cluster/22-image figure above is the
deduplicated count), **4 straddle the `image_random` split's fold
boundaries; 3 straddle `patient_group`'s.** Materially the same order of
magnitude for both -- confirms directly what the design argument already
predicted: patient-grouped splitting has no mechanism to catch a
duplicate filed under two *different* patient IDs, since it only keeps a
single declared ID's images together. This class of leakage is dedupe's
job specifically, not splitting's.

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
