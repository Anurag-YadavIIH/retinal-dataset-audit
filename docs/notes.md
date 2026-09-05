# Working notes

Scratch space. Record decisions here as they are made so WALKTHROUGH.md can be
written from evidence rather than reconstructed from memory.

## Log

- [ ] Session 1: scaffold created, ingest + split + A/B experiment.

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
