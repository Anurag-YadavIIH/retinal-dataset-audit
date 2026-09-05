# Working notes

Scratch space. Record decisions here as they are made so WALKTHROUGH.md can be
written from evidence rather than reconstructed from memory.

## Log

- [ ] Session 1: scaffold created, ingest + split + A/B experiment.

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
