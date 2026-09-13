"""Item 2 (roadmap): cross-camera domain-shift audit.

ODIR-5K mixes Canon, Zeiss and Kowa across multiple Chinese centres, and
there is no explicit camera/site column in the metadata. Raw image
resolution (before this project's own preprocessing resizes everything
to 512x512) is used as a PROXY for site -- stated once here, applies to
every number below: if several cameras happen to share a resolution,
this proxy UNDER-counts real sites. It is a lower bound on how many
distinct sources exist, not a count of cameras.

The test: train a small classifier to predict the resolution-cluster
("putative site") from the image. Critically, this trains on the
PREPROCESSED images (already resized to a common 512x512), not the raw
ones -- if the model can still recover the source after the resize, the
signal is in the optics/colour rendition, not pixel dimensions, which is
the interesting claim. Predicting resolution from an image that still
has its original resolution would be trivial and meaningless.

Then: does site correlate with diagnosis? If certain centres contribute
disproportionately many of one disease, a model can score well by
learning centre instead of pathology -- which would mean patient-level
splitting (this project's whole design) is necessary but not sufficient.

No new data needed -- everything here is derived from the raw
`Training Images/` folder and the existing manifest/labels already on
disk.

Run: python notebooks/domain_shift_audit.py
Output: printed report + artifacts/domain_shift_audit.json
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

ARTIFACTS = REPO_ROOT / "artifacts"
RAW_DIR = REPO_ROOT / "data" / "odir5k" / "ODIR-5K" / "ODIR-5K" / "Training Images"
CATEGORIES = ["N", "D", "G", "C", "A", "H", "M", "O"]
SEED = 42

# DBSCAN eps in raw pixels: merges resolutions that differ by less than
# this in each dimension into one putative-site cluster. Chosen by
# inspection (see docs/notes.md): at eps=100, 43 raw clusters emerge, 19
# of which hold >=50 images each (96% of the dataset); the rest are rare/
# anomalous resolutions consolidated into "Other" below, since a class
# with a handful of images can't be meaningfully split or evaluated.
DBSCAN_EPS = 100
MIN_CLASS_SIZE = 50


def compute_raw_resolutions(manifest: pd.DataFrame) -> pd.Series:
    """Raw (pre-preprocessing) resolution for every manifest image, by
    matching filename against the raw Training Images/ folder -- confirmed
    directly (not assumed) that every one of the 6392 manifest filenames
    has a same-named raw counterpart."""
    sizes = {}
    for p in manifest["image_path"]:
        name = Path(p).name
        with Image.open(RAW_DIR / name) as im:
            sizes[p] = im.size
    return manifest["image_path"].map(sizes)


def cluster_sites(resolutions: pd.Series) -> tuple[pd.Series, dict]:
    """DBSCAN-cluster the distinct resolutions, then consolidate clusters
    smaller than MIN_CLASS_SIZE into one "Other" class.

    Returns (site_label per image, report dict with full cluster detail
    for transparency about the consolidation).
    """
    distinct = sorted(set(resolutions))
    points = np.array(distinct)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=1).fit(points)
    res_to_raw_cluster = {res: int(lbl) for res, lbl in zip(distinct, db.labels_, strict=True)}

    raw_cluster_of = resolutions.map(res_to_raw_cluster)
    raw_sizes = raw_cluster_of.value_counts()

    big_clusters = raw_sizes[raw_sizes >= MIN_CLASS_SIZE].index.tolist()
    # Order big clusters by size (descending) for stable, readable labels.
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


def patient_level_site(manifest: pd.DataFrame) -> pd.DataFrame:
    """Assign each patient one site label (the mode of their eyes' image-
    level labels) and report how often a patient's own eyes disagree --
    checked directly, not assumed uniform."""
    per_patient = manifest.groupby("patient_id")["site_label"].agg(lambda s: s.mode().iloc[0])
    disagreement = manifest.groupby("patient_id")["site_label"].nunique()
    n_multi_eye = int((disagreement > 1).sum())
    logger.info(
        "%d/%d multi-eye patients have eyes assigned to different site clusters "
        "(expected: raw resolution is a per-eye-capture property, not guaranteed "
        "identical even within one patient's session)",
        n_multi_eye,
        int((manifest.groupby("patient_id").size() > 1).sum()),
    )
    return per_patient.rename("site_label").reset_index()


def _group_holdout(df: pd.DataFrame, holdout_frac: float, group_col: str, seed: int):
    from sklearn.model_selection import GroupKFold

    n_groups = df[group_col].nunique()
    n_splits = max(2, min(round(1.0 / holdout_frac), n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    keep_pos, holdout_pos = next(gkf.split(df, groups=df[group_col]))
    return df.iloc[keep_pos].reset_index(drop=True), df.iloc[holdout_pos].reset_index(drop=True)


def build_split(manifest: pd.DataFrame, seed: int) -> dict:
    """Patient-grouped train/val/test split on site_label -- grouped so a
    patient's fellow eye (99% of the time the same site, checked above)
    can't hand the classifier a trivial shortcut, the same leakage
    concern this project's whole pipeline is built around, applied here
    to a different target label."""
    manifest = manifest.reset_index(drop=True)
    train_val, test = _group_holdout(manifest, 0.2, "patient_id", seed)
    train, val = _group_holdout(train_val, 0.125, "patient_id", seed)
    return {
        "train": train["image_path"].tolist(),
        "val": val["image_path"].tolist(),
        "test": test["image_path"].tolist(),
    }


def fold_site_balance(manifest: pd.DataFrame, split: dict) -> dict:
    """Per-fold image count for every site class, and each class's
    train-fraction -- the check for whether the classifier could be
    reading fold structure rather than optics. build_split groups by
    patient_id only (not stratified on site_label), so an imbalance
    here would be a real, checkable risk, not assumed away."""
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
            logger.warning(
                "Site class %r has ZERO images in the test fold -- its recall "
                "cannot be evaluated and majority-baseline/accuracy numbers "
                "should be read with that in mind.",
                c,
            )

    return {"counts_per_fold": table, "per_class_train_fraction": per_class_train_fraction}


def train_site_classifier(manifest: pd.DataFrame, split: dict, class_names: list[str]) -> dict:
    """ResNet18, new head, predicting site_label from the PREPROCESSED
    (512x512-resized) image -- not the raw one. If it succeeds here, the
    signal survived the resize to a common size, so it can't be pixel
    dimensions; it has to be optics/colour rendition."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
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

    class SiteDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]
            with Image.open(path) as im:
                image = self.transform(im.convert("RGB"))
            label = class_to_idx[label_of_path.loc[path]]
            return image, label

    train_loader = DataLoader(
        SiteDataset(split["train"], transform_train), batch_size=32, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        SiteDataset(split["val"], transform_eval), batch_size=32, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        SiteDataset(split["test"], transform_eval), batch_size=32, shuffle=False, num_workers=0
    )

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


def site_vs_diagnosis(patient_sites: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """Contingency table + chi-square: does site correlate with the
    primary binary label, and (supplementary) with each of the 8
    individual disease categories -- named because certain categories
    (diabetic retinopathy, glaucoma) are the concrete concern, not an
    abstract one.

    The binary label used here is derived from the patient-level `N`
    flag (any diagnosed abnormality at all), not the per-eye `label`
    column this project's main task trains on -- `label` disagrees
    between a patient's two eyes for 22.2% of multi-eye patients
    (docs/notes.md), so picking one eye's value for a patient-level
    question would be an arbitrary, not a principled, choice. `N` is
    genuinely patient-level (confirmed elsewhere in this project) and
    answers the question this analysis is actually asking: does this
    patient have any diagnosed abnormality, not does this specific eye.
    """
    merged = patient_sites.merge(patients, on="patient_id")
    merged["any_abnormal"] = (merged["N"] != 1).astype(int)

    table_binary = pd.crosstab(merged["site_label"], merged["any_abnormal"])
    chi2_binary, p_binary, dof_binary, _ = stats.chi2_contingency(table_binary)

    per_category = {}
    for cat in CATEGORIES:
        table = pd.crosstab(merged["site_label"], merged[cat])
        if table.shape[1] < 2:
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table)
        per_category[cat] = {"chi2": float(chi2), "p": float(p), "dof": int(dof)}

    return {
        "binary_label_contingency_table": table_binary.to_dict(),
        "binary_label_chi2": float(chi2_binary),
        "binary_label_p": float(p_binary),
        "binary_label_dof": int(dof_binary),
        "per_category_chi2": per_category,
    }


def main() -> None:
    manifest = pd.read_parquet(ARTIFACTS / "manifest.parquet")

    logger.info("Computing raw resolutions for %d images...", len(manifest))
    resolutions = compute_raw_resolutions(manifest)
    site_label, cluster_report = cluster_sites(resolutions)
    manifest = manifest.assign(site_label=site_label)

    class_names = sorted(manifest["site_label"].unique(), key=lambda c: (c == "Other", c))
    logger.info("Site clusters: %s", cluster_report)

    # Persisted so splits.site_group_split (arm E, roadmap item 2
    # follow-up) can group on site without recomputing the DBSCAN
    # clustering -- the same "split once, persist it" discipline as the
    # rest of this project's splits.
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

    # Checkpoint the expensive part (the actual training run) before the
    # cheap diagnosis-correlation analysis, which has no GPU cost -- a bug
    # there should never mean re-training from scratch to see the result again.
    checkpoint_path = ARTIFACTS / "domain_shift_audit_classifier_only.json"
    checkpoint = {"cluster_report": cluster_report, "classifier_result": train_result}
    with open(checkpoint_path, "w") as fh:
        json.dump(checkpoint, fh, indent=2, default=str)
    logger.info("Checkpointed classifier result to %s", checkpoint_path)

    # The canonical manifest schema (image_path, patient_id, eye, age, sex,
    # label, dataset_name) doesn't carry the original N/D/G/C/A/H/M/O
    # one-hot flags -- those only exist in the raw full_df.csv, patient-
    # level (duplicated identically across a patient's rows).
    raw = pd.read_csv(REPO_ROOT / "data" / "odir5k" / "full_df.csv")
    raw["patient_id"] = raw["ID"].astype(str)
    patients = raw.drop_duplicates("patient_id")[["patient_id", *CATEGORIES]]
    patient_sites = patient_level_site(manifest)
    diagnosis_report = site_vs_diagnosis(patient_sites, patients)

    result = {
        "cluster_report": cluster_report,
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
    print(f"\nTest accuracy: {train_result['test_accuracy']:.4f} "
          f"(majority-class baseline: {train_result['majority_class_baseline']:.4f})")
    print(f"\nSite vs any-abnormal (patient-level N flag): "
          f"chi2={diagnosis_report['binary_label_chi2']:.2f} "
          f"p={diagnosis_report['binary_label_p']:.6f} dof={diagnosis_report['binary_label_dof']}")
    print("\nSite vs individual category:")
    for cat, r in diagnosis_report["per_category_chi2"].items():
        print(f"  {cat}: chi2={r['chi2']:.2f} p={r['p']:.6f}")


if __name__ == "__main__":
    main()
