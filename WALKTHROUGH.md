# Walkthrough

> Written for a reader who has to defend this project in a technical interview.
> Claude Code fills each section as the corresponding module is built.
> **Read this file before reading the code.**

## 1. The claim

*What is this project asserting, in one paragraph, and what evidence supports it?*

## 2. Dataset and label derivation

*Why ODIR-5K. How patient-level labels were mapped onto individual eyes, and
what that mapping costs.*

## 3. Manifest design

*Why a single canonical schema, and why patient_id is the load-bearing column.*

## 4. Splitting

*image_random vs patient_group. Why StratifiedGroupKFold. Why perfect
stratification is impossible under a group constraint.*

## 5. Why leakage is worse in retinal imaging than elsewhere

*Bilateral disease correlation. Two eyes per patient. Repeat visits and
same-session recaptures. Compare against a domain where this is milder.*

## 6. Quality scoring

*Each metric, why it was chosen, its failure modes. Why the score is a
transparent weighted sum rather than a learned one. The resolution-dependence
of variance-of-Laplacian across mixed camera models.*

## 7. Deduplication

*Why phash and embeddings catch different things. Threshold selection and how
it was validated.*

## 8. Model and metrics

*Why ResNet18 and nothing fancier. Why AUROC alone is not enough on an
imbalanced set, and what sensitivity at 95% specificity means clinically.*

## 9. Experiment design

*Why arm sizes are matched. What is and is not controlled. What would confound
the result.*

## 10. Limitations

*Single dataset. Binary task. Patient-level labels. No external validation set.
State these plainly — an interviewer will find them anyway, and finding them
first is the better position.*

---

## Interview questions

*12 likely questions with model answers. Claude Code drafts these; rewrite them
in your own words before the interview, because an answer you cannot rephrase
is an answer you do not have.*

1. Why does patient-level splitting matter more here than in, say, chest X-ray classification?
2. Your gradability score is hand-weighted. Why not learn it?
3. How would you validate the quality thresholds without ground-truth quality labels?
4. What breaks if the same patient appears under two different IDs?
5. How would this pipeline change for a real hospital PACS instead of a Kaggle zip?
6. Horizontal flip augmentation on fundus images — safe or not, and why?
7. Your labels are patient-level but your rows are eye-level. What does that do to the metrics?
8. How would you detect a camera or site confound?
9. What would you do differently if the downstream task were segmentation rather than classification?
10. Sensitivity at 95% specificity — why that operating point?
11. If the A-vs-B gap had come out near zero, what would you conclude?
12. What is the single weakest part of this project?
