"""Item 3 (external validation): does the ODIR-5K site-recoverability
finding generalise to a second dataset, or was it specific to ODIR-5K's
own multi-centre, multi-camera-stock aggregation?

Structurally parallel to notebooks/domain_shift_audit.py (same DBSCAN-on-
resolution proxy, same ResNet18 classifier design, same chi-square
correlation test), written as a separate script rather than a shared/
parameterised one -- domain_shift_audit.py is already committed, cited by
exact number throughout docs/notes.md, README.md and WALKTHROUGH.md, and
retrofitting it into a generic multi-dataset script would risk those
citations for no benefit this project needs (this is exactly the same
call arm_e_robustness_checks.py and site4_followup.py already made:
new question, new script, same conventions).

The raw-vs-preprocessed structure ends up IDENTICAL to ODIR-5K's, and
that is load-bearing, not incidental. EyePACS ships native-resolution
images (433x289 to 5184x3456), so scripts/preprocess_eyepacs.py resizes
them once to exactly 512x512 -- the same size, and the same forced
non-aspect-preserving resize, ODIR-5K's own shipped `preprocessed_images`
already use (confirmed by inspection). The classifier below trains on
those 512x512 copies while site labels come from the NATIVE resolutions,
mirroring ODIR-5K's raw/preprocessed split exactly.

Training on the native images instead would have quietly invalidated the
whole experiment: the DataLoader's Resize((256,256)) squashes a 5184x3456
image and a 2560x1920 image by different aspect-ratio factors, so the
network could read source resolution off the distortion itself rather
than off optics/colour rendition -- the trivial result this design exists
to rule out. Normalising to square 512x512 first removes that channel,
exactly as ODIR-5K's preprocessing already had.

One genuine structural difference from ODIR-5K remains:

- No independent patient-level diagnosis source. ODIR-5K ships a genuine
   patient-level `N` flag, separate from the per-eye `label`/keywords
   used for the main task -- `site_vs_diagnosis` there uses `N` precisely
   because it isn't derived from the eye-level signal being modelled.
   EyePACS has no such second source: `level` (and therefore `label`) is
   graded per eye, full stop. The patient-level diagnosis question here
   is therefore necessarily a derived one -- "was EITHER eye referable
   DR" (max over a patient's two labels) -- documented as a real
   difference in what's being tested, not glossed over as equivalent.

Run: python notebooks/domain_shift_audit_eyepacs.py
Output: printed report + artifacts/eyepacs/domain_shift_audit.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from retinaprep.utils import get_logger, set_global_seed  # noqa: E402

logger = get_logger(__name__)

# Manifest/outputs for the 512x512 copies the classifier trains on...
ARTIFACTS = REPO_ROOT / "artifacts" / "eyepacs_512"
# ...while the site proxy is read from the native-resolution originals,
# the same raw/preprocessed separation domain_shift_audit.py uses.
RAW_DIR = Path("D:/retinaprep_data/eyepacs/train")
# Reading 35,126 native JPEG headers takes ~22 minutes, so the native-
# resolution ingest's own cache is reused when present rather than paid
# for twice.
RESOLUTION_CACHE = REPO_ROOT / "artifacts" / "eyepacs" / "_resolutions_cache.parquet"
SEED = 42

# Same DBSCAN mechanics as domain_shift_audit.py; eps/min_class_size are
# NOT assumed to transfer unchanged -- reported as found (see main()'s
# cluster_report) rather than forced to reproduce ODIR-5K's 20-class shape.
DBSCAN_EPS = 100
MIN_CLASS_SIZE = 50


def compute_raw_resolutions(manifest: pd.DataFrame) -> pd.Series:
    """Native (pre-resize) resolution for every manifest image, matched by
    filename against RAW_DIR -- the manifest itself points at the 512x512
    copies, so reading resolutions off it directly would return 512x512 for
    every row and silently destroy the entire site proxy.

    Reuses the native-resolution ingest's cached scan when available (same
    filenames, ~22 minutes of JPEG-header reads otherwise).
    """
    names = manifest["image_path"].map(lambda p: Path(p).name)

    if RESOLUTION_CACHE.exists():
        cache = pd.read_parquet(RESOLUTION_CACHE)
        by_name = {
            Path(p).name: r
            for p, r in zip(cache["image_path"], cache["resolution"], strict=True)
        }
        missing = [n for n in names if n not in by_name]
        if not missing:
            logger.info("Reusing cached native resolutions from %s", RESOLUTION_CACHE)
            return names.map(by_name).map(tuple)
        logger.warning("Resolution cache missing %d image(s); rescanning from %s",
                       len(missing), RAW_DIR)

    sizes = {}
    for name in names:
        with Image.open(RAW_DIR / name) as im:
            sizes[name] = im.size
    return names.map(sizes)


def cluster_sites(resolutions: pd.Series) -> tuple[pd.Series, dict]:
    """DBSCAN-cluster distinct resolutions, consolidate anything smaller
    than MIN_CLASS_SIZE into one "Other" class. Identical mechanics to
    domain_shift_audit.py's function of the same name."""
    distinct = sorted(set(resolutions))
    points = np.array(distinct)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=1).fit(points)
    res_to_raw_cluster = {res: int(lbl) for res, lbl in zip(distinct, db.labels_, strict=True)}

    raw_cluster_of = resolutions.map(res_to_raw_cluster)
    raw_sizes = raw_cluster_of.value_counts()

    big_clusters = raw_sizes[raw_sizes >= MIN_CLASS_SIZE].index.tolist()
    big_clusters = raw_sizes.loc[big_clusters].sort_values(ascending=False).index.tolist()
    label_of_raw_cluster = {rc: f"site_{i}" for i, rc in enumerate(big_clusters)}

    site_label = raw_cluster_of.map(lambda rc: label_of_raw_cluster.get(rc, "Other"))

    report = {
        "n_distinct_resolutions": len(distinct),
        "n_raw_clusters": int(len(raw_sizes)),
        "n_final_site_classes": int(len(big_clusters) + (1 if "Other" in site_label.values else 0)),
        "dbscan_eps": DBSCAN_EPS,
        "min_class_size": MIN_CLASS_SIZE,
        "images_per_final_class": site_label.value_counts().sort_values(ascending=False).to_dict(),
    }
    return site_label, report


def check_patient_site_consistency(manifest: pd.DataFrame, site_label: pd.Series) -> dict:
    """Report (not silently assume) whether a patient's two eyes ever land
    in different site clusters -- ODIR-5K needed a fix for 10/3358
    patients; checked directly here rather than ported unconditionally."""
    df = manifest[["patient_id"]].assign(site_label=site_label.to_numpy())
    per_patient_nunique = df.groupby("patient_id")["site_label"].nunique()
    n_inconsistent = int((per_patient_nunique > 1).sum())
    logger.info(
        "%d/%d patients have two eyes assigned to different site clusters",
        n_inconsistent,
        len(per_patient_nunique),
    )
    return {"n_patients": len(per_patient_nunique), "n_inconsistent": n_inconsistent}


def _group_holdout(df: pd.DataFrame, holdout_frac: float, group_col: str, seed: int):
    from sklearn.model_selection import GroupKFold

    n_groups = df[group_col].nunique()
    n_splits = max(2, min(round(1.0 / holdout_frac), n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    keep_pos, holdout_pos = next(gkf.split(df, groups=df[group_col]))
    return df.iloc[keep_pos].reset_index(drop=True), df.iloc[holdout_pos].reset_index(drop=True)


def build_split(manifest: pd.DataFrame, seed: int) -> dict:
    """Patient-grouped train/val/test split on site_label -- identical
    rationale to domain_shift_audit.py's build_split: group by patient so
    a fellow eye can't hand the classifier a trivial shortcut."""
    manifest = manifest.reset_index(drop=True)
    train_val, test = _group_holdout(manifest, 0.2, "patient_id", seed)
    train, val = _group_holdout(train_val, 0.125, "patient_id", seed)
    return {
        "train": train["image_path"].tolist(),
        "val": val["image_path"].tolist(),
        "test": test["image_path"].tolist(),
    }


def fold_site_balance(manifest: pd.DataFrame, split: dict) -> dict:
    """Per-fold image count for every site class, flagging any class with
    zero test images -- same check as domain_shift_audit.py's function."""
    label_of_path = manifest.set_index("image_path")["site_label"]
    table = {}
    for fold, paths in split.items():
        counts = label_of_path.loc[paths].value_counts()
        table[fold] = counts.to_dict()

    all_classes = sorted(manifest["site_label"].unique())
    per_class_train_fraction = {}
    for c in all_classes:
        counts_by_fold = {fold: table[fold].get(c, 0) for fold in split}
        total = sum(counts_by_fold.values())
        per_class_train_fraction[c] = counts_by_fold["train"] / total if total else None
        if counts_by_fold["test"] == 0:
            logger.warning("Site class %r has ZERO images in the test fold", c)

    return {"counts_per_fold": table, "per_class_train_fraction": per_class_train_fraction}


class SiteDataset:
    """Module-level (not nested in train_site_classifier, as
    domain_shift_audit.py has it) specifically so DataLoader workers can
    pickle it: Windows spawns workers rather than forking, and a class
    defined inside a function body cannot be pickled by reference.

    That detail is load-bearing, not stylistic. Measured on this dataset:
    the transform pipeline costs ~72ms/image single-threaded, which at
    24,590 training images is ~30 minutes per epoch and leaves the GPU
    idle ~90% of the time (sampled: 0/0/0/0/94/0%). Keeping the class
    nested would force num_workers=0 and make a 15-epoch run a 5-8 hour
    job on this data.
    """

    def __init__(self, paths: list[str], transform, label_of_path: pd.Series, class_to_idx: dict):
        self.paths = paths
        self.transform = transform
        self.label_of_path = label_of_path
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        return image, self.class_to_idx[self.label_of_path.loc[path]]


def train_site_classifier(
    manifest: pd.DataFrame, split: dict, class_names: list[str], num_workers: int = 4
) -> dict:
    """ResNet18 predicting site_label from the image -- same design as
    domain_shift_audit.py's function of the same name (on-the-fly resize
    to 256/crop 224, identical to this project's train.py transform)."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import models, transforms

    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    label_of_path = manifest.set_index("image_path")["site_label"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    transform_train = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def _loader(paths: list[str], transform, shuffle: bool) -> DataLoader:
        return DataLoader(
            SiteDataset(paths, transform, label_of_path, class_to_idx),
            batch_size=32,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

    train_loader = _loader(split["train"], transform_train, shuffle=True)
    val_loader = _loader(split["val"], transform_eval, shuffle=False)
    test_loader = _loader(split["test"], transform_eval, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def _eval(loader) -> tuple[float, np.ndarray, np.ndarray]:
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                all_pred.append(logits.argmax(dim=1).cpu().numpy())
                all_true.append(labels.numpy())
        pred, true = np.concatenate(all_pred), np.concatenate(all_true)
        return float((pred == true).mean()), pred, true

    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    max_epochs, patience = 15, 3
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        val_acc, _, _ = _eval(val_loader)
        elapsed = time.perf_counter() - t0
        logger.info("Epoch %d/%d: %.1fs, val acc=%.4f", epoch, max_epochs, elapsed, val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping after epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    test_acc, pred, true = _eval(test_loader)
    cm = confusion_matrix(true, pred, labels=list(range(len(class_names))))

    return {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "n_train": len(split["train"]),
        "n_val": len(split["val"]),
        "n_test": len(split["test"]),
    }


def patient_level_site(manifest: pd.DataFrame) -> pd.DataFrame:
    """One site label per patient (the mode of their two eyes' labels)."""
    per_patient = manifest.groupby("patient_id")["site_label"].agg(lambda s: s.mode().iloc[0])
    return per_patient.rename("site_label").reset_index()


def site_vs_diagnosis(manifest: pd.DataFrame) -> dict:
    """Does site correlate with diagnosis? Patient-level diagnosis here is
    necessarily derived (max over both eyes' per-eye label -- "was either
    eye referable DR"), not an independent source like ODIR-5K's `N` flag
    -- see module docstring. Single binary label only: EyePACS has one DR
    grading task, not ODIR-5K's 8 disease categories, so there is no
    per-category breakdown to run here."""
    patient_sites = patient_level_site(manifest)
    patient_diag = manifest.groupby("patient_id")["label"].max().rename("any_abnormal")
    merged = patient_sites.merge(patient_diag, on="patient_id")

    table = pd.crosstab(merged["site_label"], merged["any_abnormal"])
    chi2, p, dof, _ = stats.chi2_contingency(table)

    return {
        "contingency_table": table.to_dict(),
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
    }


def main() -> None:
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")

    logger.info("Computing resolutions for %d images...", len(manifest))
    resolutions = compute_raw_resolutions(manifest)
    site_label, cluster_report = cluster_sites(resolutions)
    consistency_report = check_patient_site_consistency(manifest, site_label)
    manifest = manifest.assign(site_label=site_label)

    class_names = sorted(manifest["site_label"].unique(), key=lambda c: (c == "Other", c))
    logger.info("Site clusters: %s", cluster_report)

    site_labels_path = ARTIFACTS / "site_labels.parquet"
    manifest[["image_path", "site_label"]].to_parquet(site_labels_path, index=False)
    logger.info("Wrote %s", site_labels_path)

    split = build_split(manifest[["image_path", "patient_id", "site_label"]], SEED)
    n_train, n_val, n_test = len(split["train"]), len(split["val"]), len(split["test"])
    logger.info("Split sizes: train=%d val=%d test=%d", n_train, n_val, n_test)

    balance_report = fold_site_balance(manifest, split)
    logger.info("Per-fold site-class balance: %s", balance_report["counts_per_fold"])

    train_result = train_site_classifier(manifest, split, class_names)

    test_labels = manifest.set_index("image_path").loc[split["test"], "site_label"]
    majority_frac = test_labels.value_counts(normalize=True).iloc[0]
    train_result["majority_class_baseline"] = float(majority_frac)

    checkpoint_path = ARTIFACTS / "domain_shift_audit_classifier_only.json"
    checkpoint = {"cluster_report": cluster_report, "classifier_result": train_result}
    with open(checkpoint_path, "w") as fh:
        json.dump(checkpoint, fh, indent=2, default=str)
    logger.info("Checkpointed classifier result to %s", checkpoint_path)

    diagnosis_report = site_vs_diagnosis(manifest)

    result = {
        "cluster_report": cluster_report,
        "patient_site_consistency": consistency_report,
        "fold_site_balance": balance_report,
        "classifier_result": train_result,
        "diagnosis_report": diagnosis_report,
    }
    out_path = ARTIFACTS / "domain_shift_audit.json"
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(f"\nWrote {out_path}")
    print(f"\nSite clusters: {cluster_report['n_raw_clusters']} raw, "
          f"{cluster_report['n_final_site_classes']} after consolidating rare ones into 'Other'")
    print(f"Images per final class: {cluster_report['images_per_final_class']}")
    print(f"\nPatient/site consistency: {consistency_report['n_inconsistent']}/"
          f"{consistency_report['n_patients']} patients have eyes in different site clusters")
    print(f"\nTest accuracy: {train_result['test_accuracy']:.4f} "
          f"(majority-class baseline: {train_result['majority_class_baseline']:.4f})")
    print(f"\nSite vs any-abnormal (patient-level, max over both eyes): "
          f"chi2={diagnosis_report['chi2']:.2f} p={diagnosis_report['p']:.6f} "
          f"dof={diagnosis_report['dof']}")


if __name__ == "__main__":
    main()
