"""Step 10 — U-Net segmentation baseline.

Deliberately boring, same stance as train.py: a small textbook U-Net,
Dice+BCE loss, no architecture search, no augmentation beyond flips and
rotation. The data path is the contribution; the model is a measuring
instrument and should be unremarkable enough that nobody has to wonder
whether a clever trick produced the result.

Sized for the hardware it runs on (GTX 1050, 4GB): base width 32, batch
size 2 at 512x512, AMP off. Two hard-won constraints from earlier items
are carried over rather than rediscovered:

- **num_workers is small and eval loaders get none.** Each spawned worker
  is a separate Windows process that imports torch and reserves ~1.4GB
  for CUDA DLLs; on a 7.8GB machine an over-eager num_workers killed a
  completed training run at the final step (WALKTHROUGH.md §11).
- **Best weights go to disk the moment they improve.** The same run lost
  ~70 minutes of correct training because best_state lived only in RAM.

Both tasks use the identical code path at 512x512 full-image resolution,
so optic disc and vessel results differ because the tasks differ, not
because two pipelines differ.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from retinaprep.config import resolve_path
from retinaprep.mask_quality import load_binary_mask
from retinaprep.utils import get_logger, save_json, set_global_seed

logger = get_logger(__name__)

IMAGE_SIZE = 512


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    inter = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()
    return float(2 * inter / total) if total else 1.0


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(inter / union) if union else 1.0


class SegDataset(Dataset):
    """Module-level so DataLoader workers can pickle it under Windows spawn
    -- a class defined inside a function cannot be (WALKTHROUGH.md §11)."""

    def __init__(self, rows: list[dict], augment: bool):
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        with Image.open(row["image_path"]) as im:
            image = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        image = np.asarray(image, dtype=np.float32) / 255.0

        mask = load_binary_mask(row["mask_path"]).astype(np.uint8) * 255
        mask = np.asarray(
            Image.fromarray(mask).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST),
            dtype=np.float32,
        )
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:          # vertical flip only
                image, mask = image[::-1].copy(), mask[::-1].copy()
            k = np.random.randint(4)
            if k:
                image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()

        image = torch.from_numpy(image).permute(2, 0, 1)
        return image, torch.from_numpy(mask).unsqueeze(0)


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        self.d1, self.d2, self.d3, self.d4 = (
            _block(3, base), _block(base, base * 2),
            _block(base * 2, base * 4), _block(base * 4, base * 8),
        )
        self.pool = nn.MaxPool2d(2)
        self.bottom = _block(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.u4 = _block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        b = self.bottom(self.pool(c4))
        x = self.u4(torch.cat([self.up4(b), c4], 1))
        x = self.u3(torch.cat([self.up3(x), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.head(x)


def dice_bce_loss(logits, target, eps: float = 1.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1 - (num / den)).mean()


@torch.no_grad()
def predict_masks(model, rows: list[dict], device) -> list[np.ndarray]:
    """Binary prediction per row, at IMAGE_SIZE resolution."""
    model.eval()
    loader = DataLoader(SegDataset(rows, augment=False), batch_size=1, num_workers=0)
    out = []
    for images, _ in loader:
        prob = torch.sigmoid(model(images.to(device)))
        out.append(prob[0, 0].cpu().numpy() > 0.5)
    return out


def _resized_target(row: dict) -> np.ndarray:
    mask = load_binary_mask(row["mask_path"]).astype(np.uint8) * 255
    mask = np.asarray(Image.fromarray(mask).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST))
    return mask > 127


def run_segmentation(
    cfg: dict,
    *,
    epochs: int = 60,
    batch_size: int = 2,
    lr: float = 1e-3,
    num_workers: int = 2,
    run_name: str = "unet",
) -> dict:
    """Train a U-Net on a segmentation manifest and score the held-out fold."""
    set_global_seed(cfg["seed"])
    artifacts_dir = resolve_path(cfg, cfg["paths"]["artifacts"])
    manifest = pd.read_parquet(artifacts_dir / "manifest.parquet")

    with open(artifacts_dir / "splits" / "seg_split.json") as fh:
        split = json.load(fh)
    by_path = {r["image_path"]: r for r in manifest.to_dict("records")}
    rows = {k: [by_path[p] for p in v] for k, v in split.items()}
    logger.info(
        "Split sizes: train=%d val=%d test=%d",
        len(rows["train"]), len(rows["val"]), len(rows["test"]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        SegDataset(rows["train"], augment=True), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )

    weights_path = artifacts_dir / f"{run_name}_best.pt"
    best_val, bad_epochs, patience = -1.0, 0, 15
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            opt.zero_grad()
            loss = dice_bce_loss(model(images), masks)
            loss.backward()
            opt.step()
            total += loss.item()

        preds = predict_masks(model, rows["val"], device)
        val_dice = float(np.mean([
            dice_score(p, _resized_target(r)) for p, r in zip(preds, rows["val"], strict=True)
        ]))
        logger.info(
            "Epoch %d/%d: %.1fs loss=%.4f val_dice=%.4f",
            epoch, epochs, time.perf_counter() - t0, total / max(len(train_loader), 1), val_dice,
        )
        if val_dice > best_val:
            best_val, bad_epochs = val_dice, 0
            torch.save({"state_dict": model.state_dict(), "val_dice": val_dice,
                        "epoch": epoch}, weights_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info("Early stopping after epoch %d", epoch)
                break

    del train_loader
    model.load_state_dict(torch.load(weights_path, map_location=device)["state_dict"])

    preds = predict_masks(model, rows["test"], device)
    per_image = []
    for pred, row in zip(preds, rows["test"], strict=True):
        target = _resized_target(row)
        per_image.append({
            "image_path": row["image_path"],
            "dice": dice_score(pred, target),
            "iou": iou_score(pred, target),
        })
    df = pd.DataFrame(per_image)
    d = df["dice"]
    se = d.std(ddof=1) / np.sqrt(len(d))
    result = {
        "run_name": run_name,
        "n_train": len(rows["train"]), "n_val": len(rows["val"]), "n_test": len(rows["test"]),
        "best_val_dice": best_val,
        "test_dice_mean": float(d.mean()), "test_dice_std": float(d.std(ddof=1)),
        "test_dice_se": float(se),
        "test_dice_ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
        "test_dice_min": float(d.min()), "test_dice_max": float(d.max()),
        "test_iou_mean": float(df["iou"].mean()),
    }
    df.to_parquet(artifacts_dir / f"{run_name}_per_image.parquet", index=False)
    save_json(result, artifacts_dir / f"{run_name}_result.json")
    logger.info("TEST Dice %.4f +/- %.4f (n=%d)", result["test_dice_mean"], result["test_dice_std"],
                len(df))
    return result
