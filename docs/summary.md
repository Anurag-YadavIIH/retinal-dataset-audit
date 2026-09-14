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

There turn out to be **three separate leaks, not one, each hiding behind
the level above it once that level is fixed.**

**1. Splitting photographs randomly, without regard to which patient they
belong to, is the wrong way to evaluate a model — and it's the default way
many pipelines are built.** Over 40% of patients here had one photograph in
training and another in the test set, so the model could effectively be
tested on patients it had already partly seen. **This inflates measured
accuracy, but by less than you'd expect**: a patient's two eyes share the
same diagnosis only about 78% of the time, not 100%, so seeing one eye gives
the model a moderately reliable hint about the other, not a guarantee.
Repeating this specific comparison with a fresh random draw shrank the gap
and left us unable to stand behind it with confidence at this scale — a real
but small effect, reported as exactly that rather than rounded up.

**2. Fixing patient-level splitting doesn't fully fix the problem — it just
exposes the next one underneath.** Even with every patient's photographs
kept together, a simple classifier can still tell which of ~20 camera/centre
clusters a photograph came from with 84% accuracy — site is recoverable from
the image itself, well past what image quality alone could explain. Worse:
which site a photograph came from is statistically entangled with the
diagnosis (a formal test rules out coincidence, p well below any reasonable
threshold). A model correctly blind to *patient* identity can still be
covertly learning *where a photograph was taken* as a shortcut to the
diagnosis, rather than the disease itself.

**3. Measuring that shortcut's cost directly, by holding out entire sites at
evaluation time, produced by far the largest and most reliable effect in this
audit.** Accuracy dropped over three times as much as the patient-level leak
above, consistently across every repeat of the experiment. We checked two
obvious objections before trusting this number: could it just be that the
held-out site had a different mix of healthy/diseased photographs (yes, it
does — but forcing both groups to the same mix barely moved the result), and
could it just be one unlucky site (checked by holding out four more sites
one at a time — most showed the same drop; one, honestly, didn't, and we say
so rather than hide it).

**4. We also found a real data-integrity problem, independent of any
splitting strategy: the same photograph filed under two different patient
records**, in eight separate instances. No amount of correct splitting
catches this, because the two records claim to be different patients — it
has to be caught by comparing images to each other directly.

## What this means for anyone training or evaluating on this kind of data

- A reported accuracy number means little without knowing how the split was
  done — and "by patient" is necessary but **not sufficient**. Ask whether
  it was also checked against which clinic, camera, or imaging centre
  contributed each photograph.
- **The single largest, most reliable risk found here is a model quietly
  learning to recognize where a photograph came from, not what disease it
  shows** — a shortcut invisible to patient-level auditing alone.
- A model validated only within one hospital's or study's imaging setup has
  not been shown to generalize to a different one — this audit measured
  exactly that gap directly, on this dataset, and found it large.
- Duplicate images filed under different patient records are a separate risk
  from splitting, and easy to miss because they look like clean data.
- A single training run, or a single held-out site, is not enough evidence
  of a real effect at this scale — this audit repeats every key comparison
  and reports the spread, not just the best-looking draw.

## What we'd recommend doing differently

1. Before trusting a reported accuracy figure, confirm the split was done by
   patient, not by image — then ask the harder question: was it also checked
   against site/centre/camera, or could the model be reading that instead?
2. Before deploying a model at a new site, budget for an honest accuracy drop
   of the size measured here, not the number reported on the original
   validation split — this audit's site-holdout result is a closer analogue
   to that deployment scenario than a same-site test split is.
3. Screen for duplicate images across the whole dataset, not just within one
   declared patient's own records.
4. Re-run key comparisons more than once, across more than one held-out
   group, before calling a performance difference a real, reproducible,
   universal finding.
5. Curation removed 0.7% of images, and its effect on accuracy could not
   be measured, because the splitting procedure itself reshuffles enough
   of the training set to swamp a change that small. The quality and
   dedupe stages earn their place by catching ungradable images and
   duplicated patients, not by moving a metric.

## Bottom line

Leakage here has layers: patient-level leakage is a dial, not a switch — it
moderately inflates reported accuracy and didn't reproduce cleanly on a
second run at this scale. Site-level leakage is not a dial — it is the
biggest, most repeatable effect this audit found, and it generalizes across
most (not all) held-out sites, honestly reported either way. The clearest
finding in this audit is no longer the simplest one: don't split retinal
images randomly by photograph, split by patient — but don't stop there,
because that alone still leaves the largest leak in this dataset unaddressed.
