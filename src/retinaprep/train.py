"""Step 3 — deliberately boring ResNet18 baseline.

The model is a measuring instrument, not the contribution. Keep it fixed across
every arm so the only thing varying is the data path.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from retinaprep.config import resolve_path
from retinaprep.utils import (
    append_run_index,
    config_hash,
    get_logger,
    run_timestamp,
    save_json,
    set_global_seed,
)

logger = get_logger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_device(preference: str = "auto") -> torch.device:
    """cuda -> mps -> cpu, first available, unless a specific device is forced."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: dict) -> nn.Module:
    """torchvision resnet18, pretrained per config, new 2-class head."""
    weights = models.ResNet18_Weights.DEFAULT if cfg["train"]["pretrained"] else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def _build_transform(train_cfg: dict, augment: bool) -> transforms.Compose:
    image_size = train_cfg["image_size"]
    crop_size = train_cfg["crop_size"]
    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomCrop(crop_size))
        # Rotation only -- NOT horizontal flip. A flip turns a left eye into
        # an (anatomically wrong) right eye and destroys disc-position as a
        # real signal; see docs/notes.md.
        ops.append(transforms.RandomRotation(degrees=10))
    else:
        ops.append(transforms.CenterCrop(crop_size))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class _FundusDataset(Dataset):
    def __init__(self, image_paths: list[str], label_by_path: pd.Series, transform):
        self.image_paths = image_paths
        self.label_by_path = label_by_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        label = int(self.label_by_path.loc[path])
        return image, label


def build_dataloaders(
    cfg: dict, split: dict, manifest: pd.DataFrame | None = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Resize to cfg.train.image_size, crop to cfg.train.crop_size, ImageNet norm.

    Labels come from `manifest` (default: read fresh from
    artifacts/manifest.parquet), keyed by image_path -- pass an explicit
    curated manifest (experiment.py's build_curated_manifest) for arms whose
    pool isn't the raw canonical one.
    """
    if manifest is None:
        artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
        manifest = pd.read_parquet(artifacts_dir / "manifest.parquet")
    label_by_path = manifest.set_index("image_path")["label"]

    train_cfg = cfg["train"]
    train_tf = _build_transform(train_cfg, augment=True)
    eval_tf = _build_transform(train_cfg, augment=False)

    train_ds = _FundusDataset(split["train"], label_by_path, train_tf)
    val_ds = _FundusDataset(split["val"], label_by_path, eval_tf)
    test_ds = _FundusDataset(split["test"], label_by_path, eval_tf)

    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """AUROC, AUPRC, sensitivity at 95% specificity, confusion matrix.

    Sensitivity at high specificity is the clinically meaningful operating
    point for a screening task; accuracy on an imbalanced set is not.
    """
    device = next(model.parameters()).device
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_labels)

    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    target_fpr = 1.0 - 0.95
    idx = max(int(np.searchsorted(fpr, target_fpr, side="right")) - 1, 0)
    sensitivity_at_95_specificity = float(tpr[idx])

    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity_at_95_specificity": sensitivity_at_95_specificity,
        "confusion_matrix": cm.tolist(),  # [[tn, fp], [fn, tp]]
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def _cap_train_size(train_paths: list[str], cap: int | None, seed: int) -> list[str]:
    """Randomly (but deterministically, on the global seed) trim to `cap`.

    Matching training-set size across arms means a split-strategy artifact
    (patient_group's approximate stratification landing on a slightly
    different train count than image_random's exact one) isn't mistaken for
    an effect of the thing actually being compared. Seeded on cfg["seed"]
    rather than the run's own seed override so that two seed-reruns of the
    same arm train on the identical image set -- otherwise "seed noise"
    would be confounded with "different training images."
    """
    if cap is None or len(train_paths) <= cap:
        return train_paths
    rng = np.random.default_rng(seed)
    kept = rng.choice(len(train_paths), size=cap, replace=False)
    return [train_paths[i] for i in sorted(kept)]


def _predict_scores(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """Raw positive-class probability per example, in loader iteration
    order (matches the underlying dataset's path order exactly when the
    loader has shuffle=False, which every eval loader in this module
    does) -- for analyses that need per-example scores rather than the
    aggregate metrics `evaluate` returns."""
    device = next(model.parameters()).device
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device)
            probs = torch.softmax(model(images), dim=1)[:, 1]
            scores.append(probs.cpu().numpy())
    return np.concatenate(scores)


def run_train(
    cfg: dict,
    split_name: str | None = None,
    *,
    run_name: str | None = None,
    arm: str | None = None,
    seed_override: int | None = None,
    train_size_cap: int | None = None,
    manifest: pd.DataFrame | None = None,
    split: dict | None = None,
    save_predictions: bool = False,
    record_run: bool = True,
) -> dict:
    """Train, early-stop on val AUROC, save artifacts/runs/<name>_<ts>_<hash>/metrics.json.

    Every run gets a unique directory (name, UTC timestamp, short config
    hash) and one entry in artifacts/runs/index.json -- nothing overwrites,
    so re-running the same arm/seed keeps the earlier result on disk instead
    of silently discarding it. `run_name` controls the name prefix (defaults
    to `split_name`);
    `arm`/`seed_override`/`train_size_cap`/`manifest`/`split` are set by
    experiment.py when this is one arm of a comparison rather than a
    standalone `retinaprep train`. When `split` is omitted, it's read from
    artifacts/splits/<split_name>.json (the plain CLI path); `manifest`
    defaults to artifacts/manifest.parquet inside build_dataloaders.

    `save_predictions`, when True, also writes `<run_dir>/test_predictions.parquet`
    (image_path, true_label, score) -- model weights themselves are never
    persisted (this project doesn't need inference after the fact for its
    main pipeline), so per-example scores are the cheapest way to make a
    completed run re-analysable (e.g. a prevalence-matched re-evaluation)
    without retraining a second time for a different question.

    `record_run`, when False, skips the artifacts/runs/index.json entry --
    for a deliberate re-run of an already-reported seed (same config, same
    seed, reproducing an identical result deterministically) done only to
    capture `save_predictions` after the fact. Recording it anyway would
    silently double the seeds `load_current_run_metrics` averages over for
    that arm.
    """
    if split_name is None:
        raise SystemExit("run_train needs --split (image_random|patient_group)")

    seed = seed_override if seed_override is not None else cfg["seed"]
    set_global_seed(seed)

    train_cfg = cfg["train"]
    device = resolve_device(train_cfg.get("device", "auto"))
    logger.info("Device: %s | torch %s", device, torch.__version__)

    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    if split is None:
        split_path = artifacts_dir / "splits" / f"{split_name}.json"
        with open(split_path) as fh:
            split = json.load(fh)

    n_train_before_cap = len(split["train"])
    split = {**split, "train": _cap_train_size(split["train"], train_size_cap, cfg["seed"])}
    if len(split["train"]) != n_train_before_cap:
        logger.info(
            "Capped train set from %d to %d rows to match the smallest arm",
            n_train_before_cap,
            len(split["train"]),
        )

    train_loader, val_loader, test_loader = build_dataloaders(cfg, split, manifest=manifest)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_auroc = -1.0
    best_state = None
    epochs_without_improvement = 0
    epoch_seconds: list[float] = []

    n_epochs = train_cfg["epochs"]
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        epoch_seconds.append(time.perf_counter() - t0)

        val_metrics = evaluate(model, val_loader)
        logger.info(
            "Epoch %d/%d: %.1fs, val AUROC=%.4f",
            epoch,
            n_epochs,
            epoch_seconds[-1],
            val_metrics["auroc"],
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stop_patience"]:
                logger.info("Early stopping after epoch %d (no val AUROC improvement)", epoch)
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)

    name = run_name or split_name
    timestamp = run_timestamp()
    cfg_hash = config_hash(cfg)

    result = {
        "run_name": name,
        "arm": arm,
        "split_name": split_name,
        "seed": seed,
        "config_hash": cfg_hash,
        "timestamp": timestamp,
        "device": str(device),
        "torch_version": torch.__version__,
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
        "n_test": len(split["test"]),
        "n_epochs_run": len(epoch_seconds),
        "epoch_seconds": epoch_seconds,
        "wall_clock_seconds": sum(epoch_seconds),
        "best_val_auroc": best_val_auroc,
        "test_metrics": test_metrics,
        "config": cfg,
    }

    # Unique per run -- never overwrites an earlier run with the same
    # run_name (see utils.load_current_run_metrics for why that matters).
    run_dir = artifacts_dir / "runs" / f"{name}_{timestamp}_{cfg_hash}"
    save_json(result, run_dir / "metrics.json")
    logger.info("Wrote %s", run_dir / "metrics.json")

    if save_predictions:
        label_source = manifest if manifest is not None else pd.read_parquet(
            artifacts_dir / "manifest.parquet"
        )
        label_by_path = label_source.set_index("image_path")["label"]
        test_scores = _predict_scores(model, test_loader)
        predictions = pd.DataFrame(
            {
                "image_path": split["test"],
                "true_label": [int(label_by_path.loc[p]) for p in split["test"]],
                "score": test_scores,
            }
        )
        predictions_path = run_dir / "test_predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)
        logger.info("Wrote %s", predictions_path)

    if record_run:
        append_run_index(
            artifacts_dir,
            {
                "run_name": name,
                "arm": arm,
                "seed": seed,
                "split_name": split_name,
                "config_hash": cfg_hash,
                "timestamp": timestamp,
                "run_dir": run_dir.name,
                "n_train": result["n_train"],
                "auroc": test_metrics["auroc"],
                "auprc": test_metrics["auprc"],
                "sens_95_spec": test_metrics["sensitivity_at_95_specificity"],
            },
        )

    return result
