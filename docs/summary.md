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
dataset contained 8 such cases (1.3 per 1,000 images); the US screening
archive contained **441 (69 per 1,000)** — roughly 50 times the rate.
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

## Bottom line

Patient-level leakage is a dial, not a switch: it inflates reported accuracy
modestly and did not reproduce cleanly at our scale. Site-level leakage is
the bigger and more repeatable problem, and it survived being tested on a
second dataset from another continent — coming out stronger there. Duplicate
patient records are the third, independent problem that no splitting strategy
can solve. Don't split retinal images by photograph; split by patient — and
don't stop there, because that alone leaves the two larger leaks in place.
