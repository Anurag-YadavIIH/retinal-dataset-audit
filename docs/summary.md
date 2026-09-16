# RetinaPrep: what the audit found, and what to do about it

*For a clinical reader deciding whether to trust performance numbers from a
model trained on retinal photographs. Implementation detail is in
WALKTHROUGH.md for the engineering team.*

## The question

When a paper or vendor reports "94% accuracy" on a retinal disease-detection
model, how much is real diagnostic signal, and how much is an artifact of how
the training and test data were divided? We built a pipeline to measure this
directly, and ran it end to end on **two independent datasets**: a
multi-centre Chinese research collection (ODIR-5K, 6,392 photographs, 3,358
patients) and a US telemedicine screening archive (EyePACS, 35,126
photographs, 17,563 patients).

## What we found

Retinal datasets leak at **three levels**, and fixing one level does not fix
the next. This held on both datasets, despite different continents,
populations, cameras and collection protocols.

**1. Dividing the data photograph-by-photograph splits patients across the
line.** Around 46% of patients ended up with one eye in the training set and
the other in the test set — in *both* datasets (46.5% and 45.9%). The model
is then partly tested on patients it has already studied. This is a count,
not an estimate.

How much does that inflate the score? **Less than you might expect, and we
could not pin it down reliably.** A patient's two eyes carry the same
diagnosis about 78% of the time, not 100%, so seeing one eye is a strong hint
about the other rather than the answer. When we measured the accuracy cost it
came out small, and when we repeated the experiment it shrank further and
stopped being statistically convincing. We report both runs. The overlap is
certain; the damage it does to a headline number is not.

**2. Fixing that — splitting by patient, which is what careful teams already
do — exposes a second leak underneath: the camera or clinic.** A simple
classifier could tell which of ~14-20 imaging sources a photograph came from
with **84% accuracy on the Chinese dataset and 93% on the US one**, against a
~29% "just guess the most common one" baseline, *after* every image had been
normalised to an identical size. The signature is in the optics and colour
rendition, not the picture dimensions.

That matters because which site a photograph came from is **statistically
entangled with the diagnosis** in both datasets. A model that is correctly
blind to patient identity can still be quietly learning *where a photograph
was taken* as a shortcut to the diagnosis. When we held out entire sites at
evaluation time, accuracy dropped by roughly three times the patient-level
effect — the largest and most repeatable effect in the audit. It held up when
we forced the compared groups to have the same disease mix, and it reappeared
for 4 of 5 different held-out sites. It is not a clean per-site guarantee,
though: any *individual* site's measurement is too imprecise to stand alone.

**3. Neither split catches the same eye filed under two different patient
records — and this was far worse in the screening archive.** Comparing
equal-sized samples of 6,392 photographs each: the Chinese research
dataset contained 11 such cases (1.7 per 1,000 images, spanning eight
distinct pairs of patient records); the US screening archive contained
**441 (69 per 1,000)** — roughly 40 times the rate.
Inspected at full resolution, these show the same eye — same vessel
pattern, same optic disc — differing in colour processing rather than
being identical files.

Two honest limits on that number. It comes from a *sample* of the
screening archive, not all 35,126 photographs, so the archive-wide count
is higher but unmeasured — and it does not scale in proportion, because
the number of image pairs to compare grows with the square of the
dataset, so a 5.5x larger archive does not mean 5.5x the duplicates. A
full scan was attempted and its totals had to be discarded: at that scale
the grouping step chains merely-similar photographs into implausibly
large groups.

The important part: **splitting by patient cannot prevent this**, because the
two copies claim to be different patients. In the screening archive, 58% of
duplicate groups still spanned the training/test divide *with correct
patient-level splitting applied.*

## A fourth question: how good is the ground truth itself?

Everything above compares a model's answers against the labels. This part
asks a different question, running alongside all three levels rather than
after them: **what are the labels themselves worth?** One finding here is
the one a clinician is most likely to act on, so it comes first.

### A disc segmenter that breaks on florid retinopathy

We trained a standard model to outline the optic disc on IDRiD, a
diabetic retinopathy dataset that ships expert disc outlines. Averaged
over its 27 test images it scores 0.86 on the usual overlap measure,
which reads as respectable. **The average is misleading. Three of the 27
fail outright**, two of them at worse than 50% overlap — the model places
the disc in the wrong part of the retina, or scatters it across four or
five separate patches.

The three failures are not bad photographs. Our own image-quality module
scores them slightly *better* than the other 24 and would have rejected
none of them. What they have in common is disease. They carry about
**five and a half times the hard exudate burden** of the other images,
and two of them are the most and second-most exudate-heavy photographs in
the entire test set. The model is mistaking bright yellow-white exudates
for the optic disc, which under magnification is an entirely
understandable confusion and an entirely unacceptable one.

The clinical reading is direct. **Disc localisation degrades as exudate
burden rises — it is least reliable on the sickest eyes, which are
precisely the ones a screening programme exists to find.** An average
score across a test set will not reveal this, because florid eyes are a
minority in every test set. And disc outlining is usually not the end
product: it feeds cup-to-disc ratio, image centring, and vessel-origin
registration, so a silent failure on the most diseased images is carried
forward into whatever is computed next, where it will present as a
different problem entirely.

Two things we will not paper over. A third failure does not fit this
explanation at all — it is a relatively clean image, mid-range for
exudate, and we have no account of it. And with only three failures the
statistical support is suggestive rather than conclusive (p=0.06); the
visual evidence and the rank-1-and-2 exudate placement are doing more
work than the test is.

### Two qualified graders agree only about 78% of the time

A few public datasets have had the same photographs outlined twice, by
different people, independently. On two of them — DRIVE (20 images,
Netherlands) and CHASE_DB1 (28 images, a UK schoolchildren cohort) — we
measured how much the two graders agree with *each other* about where the
retinal vessels are. Two countries, two annotation teams, and the same
answer: about **78% overlap**.

That number is a ceiling, and it changes how a published figure should be
read. A vessel-segmentation model reported at 80% is not twenty points
short of being right; it is already at the limit of what the reference
standard can resolve. We checked this the direct way rather than
asserting it: a model trained on one grader's outlines agreed with that
grader **exactly as closely as the second human grader did** — 0.7884
against 0.7882, a difference of two ten-thousandths. On this measure, on
these 20 images, it is as close to the reference standard as a second
qualified human is — which is not the same as proving it has extracted
everything the labels contain, but it does mean a higher score here would
be hard to interpret.

A prediction we made in advance turned out to be wrong, and is reported
rather than deleted. We expected the model to absorb its own grader's
habits, and therefore to match that grader better than it matched the
other one. It did the opposite: it agreed slightly *more* with the grader
whose work it had never seen. The likely reason is that the training
objective rewards confident, well-supported markings, which pulls the
model toward the more conservative of the two humans. That explanation is
testable, and has not yet been tested.

The practical caution is that all of this rests on 20, 27 and 28 images —
that is simply how large the dual-graded public datasets are. We said so
before running anything, and nothing above depends on a significance test
that more images would have rescued.

## What this means, and what it does not

- A reported accuracy figure means little without knowing how the data was
  divided — and "by patient" is necessary but **not sufficient**.
- The largest and most reproducible risk we found is a model learning **where
  a photograph was taken** rather than what disease it shows. A model
  validated only within one imaging setup has not been shown to work in
  another, and our site-holdout result is a closer analogue to deployment at
  a new clinic than a same-site test score is.
- Duplicate records are a **separate** risk from splitting, easy to miss
  because they look like clean data, and materially more common in
  real-world screening archives than in curated research collections.
- A segmentation percentage has **no meaning without a human-vs-human
  reference** for the same task. Where we could measure one it was ~78%,
  and the model reached it.
- An average score hides the failures that matter clinically. Ours were
  concentrated on the most diseased eyes, and the image-quality module
  would not have caught a single one.
- **What we have not shown:** that any specific published model, benchmark or
  cleared product was affected. We measured contamination in a dataset, not
  anyone's results; establishing impact would require re-running each study
  with its own data division. Our duplicate count also comes from a sample of
  the screening archive, not all of it. "Site" is inferred from image
  properties, not from camera records, so it undercounts true sources.

## What we would recommend

1. Before trusting an accuracy figure, confirm the split was by **patient**,
   not by photograph — then ask the harder question: was it also checked
   against clinic, camera or imaging site?
2. Before deploying at a new site, budget for an accuracy drop of the size
   seen when whole sites are held out, not the number from the original
   validation split.
3. Screen for duplicate images across the *whole* dataset, not just within
   each declared patient record. In screening archives, assume this is
   needed until shown otherwise.
4. Re-run key comparisons more than once, and across more than one held-out
   group, before calling a difference real.
5. Be careful comparing image-quality statistics between datasets. We nearly
   reported one dataset as 24 times worse than another; almost half that gap
   turned out to be a difference in how images had been resized before
   measurement, not in the photographs themselves.
6. When a segmentation score is quoted, ask what **two humans** score against
   each other on the same task and the same images. Without that scale, a
   percentage cannot be interpreted — and a model already at the ceiling
   cannot be improved by a better model, only by better labels.
7. Ask how a model performs on the **most diseased** images specifically, not
   just on average. Where we looked, the failures clustered exactly there and
   were invisible in the mean.

## Bottom line

Patient-level leakage is a dial, not a switch: it inflates reported accuracy
modestly and did not reproduce cleanly at our scale. Site-level leakage is
the bigger and more repeatable problem, and it survived being tested on a
second dataset from another continent — coming out stronger there. Duplicate
patient records are the third, independent problem that no splitting strategy
can solve. Don't split retinal images by photograph; split by patient — and
don't stop there, because that alone leaves the two larger leaks in place.

Running underneath all three: the labels every one of these numbers is
measured against are themselves only about 78% reproducible between two
qualified humans, and a model that appears to be performing well on
average may be failing silently on the sickest eyes in the set.
