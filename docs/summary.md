# RetinaPrep: what the audit found, and what to do about it

*For a clinical reader deciding whether to trust performance numbers from a
model trained on retinal photographs. Implementation detail is in
WALKTHROUGH.md for the engineering team.*

## The question

When a paper or vendor reports "94% accuracy" on a retinal disease-detection
model, how much is real diagnostic signal, and how much is an artifact of how
the training and test data were divided? We built a pipeline on a public
dataset (ODIR-5K: 6,392 photographs, 3,358 patients) to measure this
directly, rather than assume an answer.

## What we found

**1. Splitting photographs randomly, without regard to which patient they
belong to, is the wrong way to evaluate a model — and it's the default way
many pipelines are built.** Over 40% of patients here had one photograph in
training and another in the test set, so the model could effectively be
tested on patients it had already partly seen.

**2. This inflates measured accuracy, but by less than you'd expect.**
Training the identical model on a patient-blind split versus the leaky split
showed a real gap, but a modest one. Why: a patient's two eyes share the same
diagnosis only about 78% of the time, not 100% — bilaterally symmetric
conditions like diabetic and hypertensive changes push this up, but plenty of
disease is unilateral. Seeing one eye gives the model a moderately reliable
hint about the other, not a guarantee.

**3. We also found a real data-integrity problem, independent of splitting:
the same photograph filed under two different patient records**, in eight
separate instances. No amount of correct patient-level splitting catches
this, because the two records claim to be different patients — it has to be
caught by comparing images to each other directly.

**4. Repeating the experiment with a fresh random draw shrank the gap and
left us unable to stand behind it with confidence.** That's not a
contradiction — it's what a real but small effect looks like when measured
too few times. We report both runs rather than the more favorable one.

## What this means for anyone training or evaluating on this kind of data

- A reported accuracy number means little without knowing how the split was
  done. Ask specifically whether it was done **by patient**, not by image.
- The size of this problem is task-dependent: worse for a single strongly
  bilateral condition than for the broad "any disease" task measured here,
  and worse for datasets with more repeat visits per patient than this one.
- Duplicate images filed under different patient records are a separate risk
  from splitting, and easy to miss because they look like clean data.
- One training run is not enough evidence of a real effect at this scale.

## What we'd recommend doing differently

1. Before trusting a reported accuracy figure, confirm the split was done by
   patient, not by image.
2. Screen for duplicate images across the whole dataset, not just within one
   declared patient's own records.
3. Re-run key comparisons more than once before calling a performance
   difference a real, reproducible finding.
4. Curation removed 0.7% of images, and its effect on accuracy could not
   be measured, because the splitting procedure itself reshuffles enough
   of the training set to swamp a change that small. The quality and
   dedupe stages earn their place by catching ungradable images and
   duplicated patients, not by moving a metric.

## Bottom line

The leakage problem here is a dial, not a switch: it moderately inflates
reported accuracy, it doesn't fabricate diagnostic ability out of nothing,
and it didn't reproduce cleanly on a second run at this scale. The clearest
finding in this audit is the simplest one — don't split retinal images
randomly by photograph. Split by patient, and check for duplicate patients
while you're at it.
