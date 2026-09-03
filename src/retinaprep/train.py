"""Step 3 — deliberately boring ResNet18 baseline.

The model is a measuring instrument, not the contribution. Keep it fixed across
every arm so the only thing varying is the data path.
"""

from __future__ import annotations


def build_model(cfg: dict):
    """TODO(claude-code): torchvision resnet18, pretrained, new 2-class head."""
    raise NotImplementedError


def build_dataloaders(cfg: dict, split: dict):
    """TODO(claude-code): resize to cfg.train.image_size, centre crop, ImageNet
    normalisation. Light augmentation only (flip, small rotation). Do NOT
    horizontally flip without thinking — it turns a left eye into a right eye,
    and disc position is a real anatomical cue. Discuss this in WALKTHROUGH.md."""
    raise NotImplementedError


def evaluate(model, loader) -> dict:
    """TODO(claude-code): AUROC, AUPRC, sensitivity at 95% specificity,
    confusion matrix. Sensitivity at high specificity is the clinically
    meaningful operating point; accuracy on an imbalanced set is not."""
    raise NotImplementedError


def run_train(cfg: dict, split_name: str | None = None) -> None:
    """TODO(claude-code): train, early-stop on val AUROC, save
    artifacts/runs/<name>/metrics.json including the seed and resolved config."""
    raise NotImplementedError
