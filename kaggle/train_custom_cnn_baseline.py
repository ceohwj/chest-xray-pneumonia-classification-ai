"""Train the Custom CNN baseline on Kaggle.

Project framing:
- Research/education portfolio project only.
- Not a clinical diagnostic system.
- Label mapping: 0 = NORMAL, 1 = PNEUMONIA.
- Uses duplicate-aware train/validation split files prepared before training.

How to use on Kaggle:
1. Upload or attach a Kaggle dataset containing:
   - outputs/train_split.csv
   - outputs/val_split.csv
   - data/images/
2. Edit DATA_ROOT below so it matches your Kaggle dataset mount.
3. Run this script/notebook cell on a Kaggle GPU.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 1. Configurable Kaggle paths
# =============================================================================

DATA_ROOT = "/kaggle/input/<DATASET_NAME>"
WORK_DIR = "/kaggle/working"

TRAIN_SPLIT_CSV = f"{DATA_ROOT}/outputs/train_split.csv"
VAL_SPLIT_CSV = f"{DATA_ROOT}/outputs/val_split.csv"
IMAGE_DIR = f"{DATA_ROOT}/data/images"

OUTPUT_DIR = f"{WORK_DIR}/custom_cnn_baseline"


CONFIG = {
    "seed": 42,
    "label_mapping": {"0": "NORMAL", "1": "PNEUMONIA"},
    "train_split_csv": TRAIN_SPLIT_CSV,
    "val_split_csv": VAL_SPLIT_CSV,
    "image_dir": IMAGE_DIR,
    "output_dir": OUTPUT_DIR,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 25,
    "num_workers": 2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.4,
    "optimizer": "AdamW",
    "loss": "CrossEntropyLoss",
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "early_stopping_metric": "val_f1",
    "early_stopping_patience": 5,
    "best_checkpoint_metric": "val_f1",
    "use_class_weights": True,
}


# =============================================================================
# 2. Reproducibility and paths
# =============================================================================


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "checkpoints": output_dir / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


# =============================================================================
# 3. Transforms
# =============================================================================


class TrainTransform:
    """Safe, mild augmentations for chest X-ray baseline training."""

    def __init__(
        self,
        image_size: int = 224,
        rotation_degrees: float = 7.0,
        translate_frac: float = 0.04,
        brightness: float = 0.08,
        contrast: float = 0.08,
    ) -> None:
        self.image_size = image_size
        self.rotation_degrees = rotation_degrees
        self.translate_frac = translate_frac
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        max_shift = int(self.image_size * self.translate_frac)
        tx = random.randint(-max_shift, max_shift)
        ty = random.randint(-max_shift, max_shift)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, translate=(tx, ty), fillcolor=0)

        brightness_factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        contrast_factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        return pil_to_tensor(image)


class ValTransform:
    """Deterministic validation transform."""

    def __init__(self, image_size: int = 224) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return pil_to_tensor(image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


# =============================================================================
# 4. Dataset and DataLoader
# =============================================================================


class ChestXrayDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_dir: str | Path, transform=None) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform

        required = {"file_name", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.csv_path}: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, file_name: str) -> Path:
        raw = Path(str(file_name))
        candidates = [
            self.image_dir / raw,
            self.image_dir / raw.name,
            self.image_dir / "train" / raw.name,
            self.image_dir / "test" / raw.name,
            self.image_dir.parent / raw,
            self.image_dir.parent / "images" / raw,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve image path for file_name={file_name}; tried {candidates[:4]}")

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = self.resolve_path(row["file_name"])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label, str(row["file_name"])


def make_loaders(config: dict[str, Any], device: torch.device) -> tuple[DataLoader, DataLoader, ChestXrayDataset, ChestXrayDataset]:
    train_dataset = ChestXrayDataset(
        config["train_split_csv"],
        config["image_dir"],
        transform=TrainTransform(image_size=config["image_size"]),
    )
    val_dataset = ChestXrayDataset(
        config["val_split_csv"],
        config["image_dir"],
        transform=ValTransform(image_size=config["image_size"]),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset, val_dataset


# =============================================================================
# 5. Model
# =============================================================================


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# 6. Metrics
# =============================================================================


def compute_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = train_df["label"].value_counts().sort_index()
    total = counts.sum()
    weights = []
    for label in [0, 1]:
        count = int(counts.get(label, 0))
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total / (2.0 * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pneumonia: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity_pneumonia_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity_normal_recall": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob_pneumonia))
    except ValueError as exc:
        metrics["auroc"] = None
        metrics["auroc_reason"] = str(exc)
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob_pneumonia))
    except ValueError as exc:
        metrics["pr_auc"] = None
        metrics["pr_auc_reason"] = str(exc)
    return metrics


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    val_df: pd.DataFrame,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    records = []

    with torch.no_grad():
        for images, labels, file_names in loader:
            images = images.to(device)
            labels = labels.long().to(device)
            logits = model(images)
            per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
            loss = per_sample_loss.mean()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

            for i in range(batch_size):
                prob_normal = float(probs[i, 0].detach().cpu().item())
                prob_pneumonia = float(probs[i, 1].detach().cpu().item())
                pred_label = int(preds[i].detach().cpu().item())
                true_label = int(labels[i].detach().cpu().item())
                records.append(
                    {
                        "file_name": file_names[i],
                        "true_label": true_label,
                        "true_class": "NORMAL" if true_label == 0 else "PNEUMONIA",
                        "pred_label": pred_label,
                        "pred_class": "NORMAL" if pred_label == 0 else "PNEUMONIA",
                        "probability_NORMAL": prob_normal,
                        "probability_PNEUMONIA": prob_pneumonia,
                        "confidence": max(prob_normal, prob_pneumonia),
                        "loss": float(per_sample_loss[i].detach().cpu().item()),
                    }
                )

    pred_df = pd.DataFrame(records)
    val_loss = total_loss / max(total_items, 1)
    metrics = compute_metrics(
        pred_df["true_label"].to_numpy(),
        pred_df["pred_label"].to_numpy(),
        pred_df["probability_PNEUMONIA"].to_numpy(),
    )

    meta_cols = [col for col in ["file_name", "duplicate_group_id", "class_name"] if col in val_df.columns]
    if meta_cols:
        pred_df = pred_df.merge(val_df[meta_cols], on="file_name", how="left", suffixes=("", "_split"))
    return val_loss, metrics, pred_df


# =============================================================================
# 7. Training
# =============================================================================


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels, _file_names in loader:
        images = images.to(device)
        labels = labels.long().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config: dict[str, Any],
    val_metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "config": config,
        "validation_metrics": to_jsonable(val_metrics),
        "label_mapping": {0: "NORMAL", 1: "PNEUMONIA"},
        "seed": config["seed"],
    }
    torch.save(checkpoint, path)


def train_baseline() -> dict[str, Any]:
    set_seed(CONFIG["seed"])
    paths = prepare_output_dirs(CONFIG["output_dir"])
    save_json(CONFIG, paths["root"] / "config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, train_dataset, val_dataset = make_loaders(CONFIG, device)
    train_df = train_dataset.df.copy()
    val_df = val_dataset.df.copy()
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print("Train class counts:")
    print(train_df["label"].value_counts().sort_index())
    print("Validation class counts:")
    print(val_df["label"].value_counts().sort_index())

    model = CustomCNN(num_classes=2, dropout=CONFIG["dropout"]).to(device)
    param_count = count_parameters(model)
    print(f"Trainable parameters: {param_count:,}")

    class_weights = compute_class_weights(train_df, device) if CONFIG["use_class_weights"] else None
    print(f"Class weights: {class_weights.detach().cpu().tolist() if class_weights is not None else None}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CONFIG["scheduler_factor"],
        patience=CONFIG["scheduler_patience"],
    )

    best_f1 = -math.inf
    best_epoch = None
    best_metrics = None
    best_pred_df = None
    epochs_without_improvement = 0
    logs = []
    latest_pred_df = None

    for epoch in range(1, CONFIG["epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, pred_df = validate(model, val_loader, criterion, device, val_df)
        scheduler.step(val_metrics["f1"])

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "sensitivity_pneumonia_recall": val_metrics["sensitivity_pneumonia_recall"],
            "specificity_normal_recall": val_metrics["specificity_normal_recall"],
            "f1": val_metrics["f1"],
            "auroc": val_metrics.get("auroc"),
            "pr_auc": val_metrics.get("pr_auc"),
            "fn": val_metrics["fn"],
            "fp": val_metrics["fp"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - start,
        }
        logs.append(log_row)
        latest_pred_df = pred_df

        print(
            f"Epoch {epoch:02d}/{CONFIG['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"f1={val_metrics['f1']:.4f} sens={val_metrics['sensitivity_pneumonia_recall']:.4f} "
            f"spec={val_metrics['specificity_normal_recall']:.4f} auroc={val_metrics.get('auroc')}"
        )

        save_checkpoint(
            paths["checkpoints"] / "latest_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            CONFIG,
            val_metrics | {"val_loss": val_loss, "train_loss": train_loss},
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_metrics = val_metrics | {"val_loss": val_loss, "train_loss": train_loss, "epoch": epoch}
            best_pred_df = pred_df.copy()
            epochs_without_improvement = 0
            save_checkpoint(
                paths["checkpoints"] / "best_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                CONFIG,
                best_metrics,
            )
        else:
            epochs_without_improvement += 1

        pd.DataFrame(logs).to_csv(paths["root"] / "train_log.csv", index=False)
        if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_metrics is None or best_pred_df is None:
        raise RuntimeError("Training finished without best metrics. Check data and metric computation.")

    final_metrics = logs[-1].copy()
    final_metrics["confusion_matrix"] = val_metrics["confusion_matrix"]
    final_metrics["best_epoch"] = best_epoch
    save_json(best_metrics, paths["root"] / "best_metrics.json")
    save_json(final_metrics, paths["root"] / "final_metrics.json")

    save_confusion_outputs(best_metrics, best_pred_df, paths["root"])
    plot_all_figures(pd.DataFrame(logs), best_pred_df, best_metrics, paths["figures"])
    write_markdown_report(
        output_path=paths["root"] / "custom_cnn_baseline_report.md",
        config=CONFIG,
        train_df=train_df,
        val_df=val_df,
        param_count=param_count,
        logs=pd.DataFrame(logs),
        best_metrics=best_metrics,
        final_metrics=final_metrics,
        best_epoch=best_epoch,
        class_weights=class_weights.detach().cpu().tolist() if class_weights is not None else None,
    )

    return {
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "output_dir": str(paths["root"]),
        "latest_predictions": latest_pred_df,
    }


# =============================================================================
# 8. Outputs, plots, and report
# =============================================================================


def save_confusion_outputs(metrics: dict[str, Any], pred_df: pd.DataFrame, output_dir: Path) -> None:
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_NORMAL_0", "actual_PNEUMONIA_1"],
        columns=["pred_NORMAL_0", "pred_PNEUMONIA_1"],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    false_negatives = pred_df[(pred_df["true_label"] == 1) & (pred_df["pred_label"] == 0)].copy()
    false_positives = pred_df[(pred_df["true_label"] == 0) & (pred_df["pred_label"] == 1)].copy()
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)
    false_positives.to_csv(output_dir / "false_positives.csv", index=False)


def plot_confusion_matrix(cm: list[list[int]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    arr = np.asarray(cm)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred NORMAL", "Pred PNEUMONIA"])
    ax.set_yticks([0, 1], labels=["Actual NORMAL", "Actual PNEUMONIA"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black")
    ax.set_title("Validation Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_train_val_loss(logs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(logs["epoch"], logs["train_loss"], marker="o", label="train_loss")
    ax.plot(logs["epoch"], logs["val_loss"], marker="o", label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train/Validation Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_curves(logs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in ["accuracy", "precision", "sensitivity_pneumonia_recall", "specificity_normal_recall", "f1", "auroc", "pr_auc"]:
        if col in logs and logs[col].notna().any():
            ax.plot(logs["epoch"], logs[col], marker="o", label=col)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation Metric Curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc_curve(pred_df: pd.DataFrame, path: Path) -> None:
    y_true = pred_df["true_label"].to_numpy()
    y_score = pred_df["probability_PNEUMONIA"].to_numpy()
    fpr, tpr, _thresholds = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUROC={auroc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Sensitivity")
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pr_curve(pred_df: pd.DataFrame, path: Path) -> None:
    y_true = pred_df["true_label"].to_numpy()
    y_score = pred_df["probability_PNEUMONIA"].to_numpy()
    precision, recall, _thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR-AUC={pr_auc:.4f}")
    ax.set_xlabel("Recall / Sensitivity")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_all_figures(logs: pd.DataFrame, pred_df: pd.DataFrame, best_metrics: dict[str, Any], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(best_metrics["confusion_matrix"], figures_dir / "confusion_matrix.png")
    plot_train_val_loss(logs, figures_dir / "train_val_loss.png")
    plot_metric_curves(logs, figures_dir / "metric_curves.png")
    try:
        plot_roc_curve(pred_df, figures_dir / "roc_curve.png")
    except ValueError as exc:
        print(f"Could not plot ROC curve: {exc}")
    try:
        plot_pr_curve(pred_df, figures_dir / "pr_curve.png")
    except ValueError as exc:
        print(f"Could not plot PR curve: {exc}")


def class_distribution_table(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    lines = ["| Label | Class | Count | Percent |", "| --- | --- | --- | --- |"]
    for label, name in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        count = int(counts.get(label, 0))
        percent = count / total * 100 if total else 0.0
        lines.append(f"| {label} | {name} | {count} | {percent:.2f}% |")
    return "\n".join(lines)


def write_markdown_report(
    output_path: Path,
    config: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    param_count: int,
    logs: pd.DataFrame,
    best_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    best_epoch: int,
    class_weights: list[float] | None,
) -> None:
    cm = best_metrics["confusion_matrix"]
    lines = [
        "# Custom CNN Baseline Report",
        "",
        "This report is for a research/education portfolio project. It does not establish clinical diagnostic validity.",
        "",
        "## 1. Purpose",
        "",
        "This is the first formal Custom CNN baseline after dataset audit, duplicate-aware split creation, and smoke testing.",
        "",
        "## 2. Dataset split",
        "",
        f"- train split size: {len(train_df)}",
        f"- validation split size: {len(val_df)}",
        "- duplicate-aware split status: no duplicate_group_id should cross train/validation in the prepared split.",
        "- patient-level leakage limitation: patient identifiers are unavailable, so patient-level leakage cannot be fully excluded.",
        "",
        "### Train class distribution",
        "",
        class_distribution_table(train_df),
        "",
        "### Validation class distribution",
        "",
        class_distribution_table(val_df),
        "",
        "## 3. Model architecture",
        "",
        f"- input shape: [batch_size, 3, {config['image_size']}, {config['image_size']}]",
        "- output shape: [batch_size, 2]",
        "- CNN blocks: 4 x Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d",
        "- pooling: AdaptiveAvgPool2d",
        f"- dropout: {config['dropout']}",
        f"- trainable parameter count: {param_count:,}",
        "",
        "## 4. Training setup",
        "",
        f"- seed: {config['seed']}",
        f"- image size: {config['image_size']}",
        f"- batch size: {config['batch_size']}",
        f"- epochs configured: {config['epochs']}",
        f"- epochs run: {int(logs['epoch'].max())}",
        f"- optimizer: {config['optimizer']}",
        f"- learning rate: {config['learning_rate']}",
        f"- scheduler: {config['scheduler']}",
        f"- loss function: {config['loss']}",
        f"- class weighting: {class_weights}",
        "- augmentation: resize, mild rotation, small translation, brightness/contrast adjustment; no vertical flip.",
        "",
        "## 5. Validation metrics",
        "",
        f"- best epoch by validation F1-score: {best_epoch}",
        f"- best validation loss: {best_metrics.get('val_loss')}",
        f"- best accuracy: {best_metrics.get('accuracy')}",
        f"- best precision: {best_metrics.get('precision')}",
        f"- best sensitivity / PNEUMONIA recall: {best_metrics.get('sensitivity_pneumonia_recall')}",
        f"- best specificity / NORMAL recall: {best_metrics.get('specificity_normal_recall')}",
        f"- best F1-score: {best_metrics.get('f1')}",
        f"- best AUROC: {best_metrics.get('auroc')}",
        f"- best PR-AUC: {best_metrics.get('pr_auc')}",
        f"- final epoch metrics: {json.dumps(to_jsonable(final_metrics), ensure_ascii=False)}",
        "",
        "## 6. Confusion matrix",
        "",
        "| Actual \\ Predicted | NORMAL | PNEUMONIA |",
        "| --- | --- | --- |",
        f"| NORMAL | {cm[0][0]} | {cm[0][1]} |",
        f"| PNEUMONIA | {cm[1][0]} | {cm[1][1]} |",
        "",
        "## 7. FN/FP review",
        "",
        f"- FN count (true PNEUMONIA predicted NORMAL): {best_metrics.get('fn')}",
        f"- FP count (true NORMAL predicted PNEUMONIA): {best_metrics.get('fp')}",
        "- Do not overinterpret individual samples without image review.",
        "",
        "## 8. Limitations",
        "",
        "- Custom CNN baseline only.",
        "- Patient-level leakage cannot be fully excluded.",
        "- Train/test near-duplicate limitation remains.",
        "- Internal validation only.",
        "- No clinical validity claim.",
        "- External validation would be required for clinical use.",
        "- Do not select a model based on accuracy alone.",
        "",
        "## 9. Next steps",
        "",
        "- ResNet18/ResNet50 transfer learning.",
        "- DenseNet121 comparison.",
        "- EfficientNet-B0/B1 comparison.",
        "- Grad-CAM analysis.",
        "- Threshold tuning on validation set.",
        "- FN/FP image review.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 9. Execute
# =============================================================================


if __name__ == "__main__":
    results = train_baseline()
    print("Training complete.")
    print(json.dumps(to_jsonable({k: v for k, v in results.items() if k != "latest_predictions"}), indent=2))
