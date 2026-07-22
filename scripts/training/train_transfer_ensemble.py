"""Train the transfer-learning ensemble in a local Python environment.

Research/education portfolio project only. This script trains binary
NORMAL/PNEUMONIA classifiers and should not be described as a clinical
diagnostic system.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset


MODEL_NAMES = ("densenet121", "convnext_tiny", "efficientnet_b3")
TTA_MODES = ("crop", "crop_flip", "full", "full_flip")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local transfer-learning ensemble.")
    parser.add_argument("--train-csv", default="data/splits/train_strict_duplicate_aware.csv")
    parser.add_argument("--val-csv", default="data/splits/val_strict_duplicate_aware.csv")
    parser.add_argument("--test-csv", default="data/test.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--output-dir", default="outputs/transfer_ensemble")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints/transfer_ensemble")
    parser.add_argument("--submission-dir", default="outputs/submissions")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-train-samples", type=int, default=96)
    parser.add_argument("--debug-val-samples", type=int, default=32)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def require_torchvision():
    try:
        from torchvision import models, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torchvision is required for transfer ensemble training.") from exc
    return models, transforms


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        transform: Any | None,
        require_label: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.require_label = require_label
        self.missing_warned: set[str] = set()

        if "file_name" not in self.df.columns:
            raise ValueError("DataFrame must contain a file_name column.")
        if require_label and "label" not in self.df.columns:
            raise ValueError("Training/validation DataFrame must contain a label column.")

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, file_name: str) -> Path | None:
        raw = Path(str(file_name))
        candidates = [
            self.image_dir / raw,
            self.image_dir / "train" / raw.name,
            self.image_dir / "test" / raw.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def load_image(self, file_name: str) -> Image.Image:
        image_path = self.resolve_path(file_name)
        if image_path is None:
            if file_name not in self.missing_warned:
                print(f"WARNING: image file not found for {file_name}; using fallback black image.")
                self.missing_warned.add(file_name)
            return Image.new("RGB", (512, 512), color=(0, 0, 0))
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        file_name = str(row["file_name"])
        image = self.load_image(file_name)
        if self.transform is not None:
            image = self.transform(image)
        if self.require_label:
            label = torch.tensor(float(row["label"]), dtype=torch.float32)
            return image, label, file_name
        return image, file_name


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        padding = kernel_size // 2
        self.spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        channel_attention = torch.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        x = x * channel_attention[:, :, None, None]
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial = torch.amax(x, dim=1, keepdim=True)
        spatial_attention = torch.sigmoid(self.spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
        return x * spatial_attention


class DenseNet121CBAM(nn.Module):
    def __init__(self, base: nn.Module, dropout: float) -> None:
        super().__init__()
        self.features = base.features
        in_features = base.classifier.in_features
        self.cbam = CBAM(in_features)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = torch.relu(features)
        features = self.cbam(features)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)


def weights_or_none(models: Any, attr_name: str, pretrained: bool) -> Any | None:
    if not pretrained:
        return None
    weight_enum = getattr(models, attr_name, None)
    if weight_enum is None:
        return None
    return getattr(weight_enum, "DEFAULT", None) or getattr(weight_enum, "IMAGENET1K_V1", None)


def instantiate_model(model_name: str, dropout: float, pretrained: bool) -> nn.Module:
    models, _transforms = require_torchvision()
    name = model_name.lower()

    def _build(weights: Any | None) -> nn.Module:
        if name == "densenet121":
            base = models.densenet121(weights=weights)
            return DenseNet121CBAM(base, dropout=dropout)
        if name == "convnext_tiny":
            model = models.convnext_tiny(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 1)
            return model
        if name == "efficientnet_b3":
            model = models.efficientnet_b3(weights=weights)
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
            return model
        raise ValueError(f"Unsupported model: {model_name}")

    weight_attrs = {
        "densenet121": "DenseNet121_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "efficientnet_b3": "EfficientNet_B3_Weights",
    }
    weights = weights_or_none(models, weight_attrs[name], pretrained)
    try:
        return _build(weights)
    except Exception as exc:
        if pretrained and weights is not None:
            print(f"WARNING: pretrained weights failed for {model_name}: {exc}. Falling back to weights=None.")
            return _build(None)
        raise


def build_transform(image_size: int, train: bool = False, tta_mode: str = "crop") -> Any:
    _models, transforms = require_torchvision()
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=7),
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if tta_mode.startswith("crop"):
        ops: list[Any] = [transforms.Resize(int(round(image_size * 1.14))), transforms.CenterCrop(image_size)]
    elif tta_mode.startswith("full"):
        ops = [transforms.Resize((image_size, image_size))]
    else:
        raise ValueError(f"Unsupported TTA mode: {tta_mode}")
    if tta_mode.endswith("flip"):
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose(ops)


def make_loader(
    df: pd.DataFrame,
    image_dir: str | Path,
    transform: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    require_label: bool,
    shuffle: bool = False,
) -> DataLoader:
    dataset = ChestXrayDataset(df, image_dir=image_dir, transform=transform, require_label=require_label)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc": auroc,
        "pneumonia_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_pos_weight(df: pd.DataFrame, device: torch.device) -> torch.Tensor | None:
    counts = df["label"].value_counts()
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    if positives == 0:
        return None
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels, _file_names in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device, require_label: bool) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        if require_label:
            images, labels, file_names = batch
        else:
            images, file_names = batch
            labels = None
        images = images.to(device)
        probs = torch.sigmoid(model(images)).view(-1).detach().cpu().numpy()
        for index, file_name in enumerate(file_names):
            row = {"file_name": file_name, "prob": float(probs[index])}
            if labels is not None:
                row["label"] = int(labels[index].item())
            rows.append(row)
    return pd.DataFrame(rows)


def predict_tta(
    model: nn.Module,
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    require_label: bool,
) -> pd.DataFrame:
    mode_dfs = []
    for mode in TTA_MODES:
        loader = make_loader(
            df,
            args.image_dir,
            build_transform(args.image_size, train=False, tta_mode=mode),
            args.batch_size,
            args.num_workers,
            device,
            require_label=require_label,
            shuffle=False,
        )
        mode_dfs.append(predict_loader(model, loader, device, require_label=require_label))

    merged = mode_dfs[0][["file_name"]].copy()
    if require_label:
        merged["label"] = mode_dfs[0]["label"].astype(int)
    probs = np.stack([mode_df["prob"].to_numpy(dtype=float) for mode_df in mode_dfs], axis=0)
    merged["prob"] = probs.mean(axis=0)
    return merged


def split_folds(df: pd.DataFrame, num_folds: int, seed: int) -> pd.DataFrame:
    if num_folds < 2:
        raise ValueError("--num-folds must be at least 2.")
    out = df.reset_index(drop=True).copy()
    labels = out["label"].to_numpy(dtype=int)
    out["fold"] = -1

    groups = out["duplicate_group_id"].astype(str).to_numpy() if "duplicate_group_id" in out.columns else None
    if groups is not None and len(np.unique(groups)) >= num_folds:
        splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = splitter.split(out, labels, groups)
    else:
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = splitter.split(out, labels)

    for fold, (_train_idx, val_idx) in enumerate(splits):
        out.loc[val_idx, "fold"] = fold
    return out


def sample_debug_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    try:
        return (
            df.groupby("label", group_keys=False)
            .apply(lambda part: part.sample(min(len(part), max(1, n // 2)), random_state=seed))
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    except ValueError:
        return df.sample(n=n, random_state=seed).reset_index(drop=True)


def train_fold_model(
    model_name: str,
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = Path(args.checkpoint_dir) / model_name / f"fold_{fold}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(
        train_df,
        args.image_dir,
        build_transform(args.image_size, train=True),
        args.batch_size,
        args.num_workers,
        device,
        require_label=True,
        shuffle=True,
    )
    model = instantiate_model(model_name, dropout=args.dropout, pretrained=args.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_df, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_pred = predict_tta(model, val_df, args, device, require_label=True)
        metrics = binary_metrics(val_pred["label"].to_numpy(dtype=int), val_pred["prob"].to_numpy(dtype=float), threshold=0.5)
        scheduler.step(0.0 if math.isnan(metrics["auroc"]) else metrics["auroc"])

        print(
            f"{model_name} fold={fold} epoch={epoch}/{args.epochs} "
            f"loss={train_loss:.4f} f1={metrics['f1']:.4f} auroc={metrics['auroc']} "
            f"pneu_recall={metrics['pneumonia_recall']:.4f} normal_recall={metrics['normal_recall']:.4f} "
            f"fn={metrics['fn']} fp={metrics['fp']} seconds={time.time() - start:.1f}"
        )
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics | {"epoch": epoch, "train_loss": train_loss}
            best_epoch = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "metrics": to_jsonable(best_metrics),
                    "label_mapping": {0: "NORMAL", 1: "PNEUMONIA"},
                },
                checkpoint_path,
            )

    if best_metrics is None:
        raise RuntimeError(f"No checkpoint produced for {model_name} fold {fold}.")
    best_metrics = best_metrics | {"model": model_name, "fold": fold, "best_epoch": best_epoch}
    return checkpoint_path, best_metrics


def load_checkpoint_model(model_name: str, checkpoint_path: Path, args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = instantiate_model(model_name, dropout=args.dropout, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def grid_search_ensemble(oof_by_model: dict[str, pd.DataFrame], active_models: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = next(iter(oof_by_model.values()))[["file_name", "label", "fold"]].copy()
    prob_matrix = {}
    for model_name in active_models:
        merged = base[["file_name"]].merge(
            oof_by_model[model_name][["file_name", "prob"]], on="file_name", how="left"
        )
        prob_matrix[model_name] = merged["prob"].to_numpy(dtype=float)

    labels = base["label"].to_numpy(dtype=int)
    best: dict[str, Any] | None = None
    thresholds = np.round(np.arange(0.20, 0.8001, 0.005), 3)

    if len(active_models) == 1:
        weight_candidates = [{active_models[0]: 1.0}]
    else:
        weight_candidates = []
        for w_densenet in np.round(np.arange(0.0, 1.0001, 0.05), 2):
            for w_convnext in np.round(np.arange(0.0, 1.0001, 0.05), 2):
                w_efficientnet = round(1.0 - float(w_densenet) - float(w_convnext), 2)
                if w_efficientnet < -1e-9:
                    continue
                weights = {
                    "densenet121": float(w_densenet),
                    "convnext_tiny": float(w_convnext),
                    "efficientnet_b3": max(0.0, float(w_efficientnet)),
                }
                if any(weights.get(name, 0.0) > 0 for name in active_models):
                    total = sum(weights.get(name, 0.0) for name in active_models)
                    if total > 0:
                        weight_candidates.append({name: weights.get(name, 0.0) / total for name in active_models})

    for weights in weight_candidates:
        ensemble_prob = sum(prob_matrix[name] * weight for name, weight in weights.items())
        for threshold in thresholds:
            score = float(f1_score(labels, (ensemble_prob >= threshold).astype(int), zero_division=0))
            if best is None or score > best["best_f1"]:
                best = {
                    "best_f1": score,
                    "best_threshold": float(threshold),
                    "best_weights": weights,
                    "prob": ensemble_prob,
                }

    if best is None:
        raise RuntimeError("Ensemble grid search failed to produce a result.")

    ensemble_df = base.copy()
    ensemble_df["prob"] = best.pop("prob")
    metrics = binary_metrics(labels, ensemble_df["prob"].to_numpy(dtype=float), best["best_threshold"])
    config = {
        **best,
        "oof_accuracy": metrics["accuracy"],
        "oof_auroc": metrics["auroc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "pneumonia_recall": metrics["pneumonia_recall"],
        "normal_recall": metrics["normal_recall"],
    }
    return ensemble_df, config


def create_submission(test_probs: pd.DataFrame, threshold: float, sample_submission: str | Path, submission_dir: str | Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_submission)
    if "file_name" not in sample.columns:
        raise ValueError("sample_submission must contain file_name column.")
    target_cols = [col for col in sample.columns if col != "file_name"]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly one target column, got {target_cols}.")
    target_col = target_cols[0]
    merged = sample[["file_name"]].merge(test_probs, on="file_name", how="left")
    if merged["prob"].isna().any():
        missing = merged.loc[merged["prob"].isna(), "file_name"].head().tolist()
        raise ValueError(f"Missing test probabilities for sample rows: {missing}")
    submission = sample[["file_name"]].copy()
    submission[target_col] = (merged["prob"].to_numpy(dtype=float) >= threshold).astype(int)
    output_path = Path(submission_dir) / "submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def create_threshold_submissions(
    test_probs: pd.DataFrame,
    sample_submission: str | Path,
    submission_dir: str | Path,
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> None:
    sample = pd.read_csv(sample_submission)
    target_cols = [col for col in sample.columns if col != "file_name"]
    if len(target_cols) != 1:
        raise ValueError(f"Expected exactly one target column, got {target_cols}.")
    target_col = target_cols[0]
    merged = sample[["file_name"]].merge(test_probs, on="file_name", how="left")
    if merged["prob"].isna().any():
        missing = merged.loc[merged["prob"].isna(), "file_name"].head().tolist()
        raise ValueError(f"Missing test probabilities for threshold submissions: {missing}")

    output_dir = Path(submission_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for threshold in thresholds:
        submission = sample[["file_name"]].copy()
        submission[target_col] = (merged["prob"].to_numpy(dtype=float) >= threshold).astype(int)
        submission.to_csv(output_dir / f"submission_threshold_{threshold:.2f}.csv", index=False)


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.debug:
        args.num_folds = 2
        args.epochs = 2
        args.num_workers = 0
        if args.image_size is None:
            args.image_size = 224
    elif args.image_size is None:
        args.image_size = 384

    for directory in [args.output_dir, args.checkpoint_dir, args.submission_dir, args.metrics_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    warnings.filterwarnings("default")
    args = prepare_args(parse_args())
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    save_json(vars(args), Path(args.output_dir) / "config.json")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    if args.debug:
        all_df = sample_debug_df(all_df, args.debug_train_samples + args.debug_val_samples, args.seed)
    folded_df = split_folds(all_df, args.num_folds, args.seed)
    folded_df.to_csv(Path(args.output_dir) / "fold_assignments.csv", index=False)

    fold_metrics: list[dict[str, Any]] = []
    oof_by_model: dict[str, pd.DataFrame] = {}
    test_prob_by_model: dict[str, np.ndarray] = {}
    test_df = pd.read_csv(args.test_csv)

    for model_name in args.models:
        model_oof: list[pd.DataFrame] = []
        model_test_probs: list[np.ndarray] = []
        for fold in sorted(folded_df["fold"].unique()):
            fold_train_df = folded_df[folded_df["fold"] != fold].copy()
            fold_val_df = folded_df[folded_df["fold"] == fold].copy()
            if args.debug:
                fold_train_df = sample_debug_df(fold_train_df, args.debug_train_samples, args.seed + int(fold))
                fold_val_df = sample_debug_df(fold_val_df, args.debug_val_samples, args.seed + int(fold))

            checkpoint_path, metrics = train_fold_model(model_name, int(fold), fold_train_df, fold_val_df, args, device)
            fold_metrics.append(metrics)
            pd.DataFrame(fold_metrics).to_csv(Path(args.metrics_dir) / "transfer_ensemble_fold_metrics.csv", index=False)

            model = load_checkpoint_model(model_name, checkpoint_path, args, device)
            val_pred = predict_tta(model, fold_val_df, args, device, require_label=True)
            val_pred["fold"] = int(fold)
            model_oof.append(val_pred[["file_name", "label", "fold", "prob"]])

            test_pred = predict_tta(model, test_df, args, device, require_label=False)
            model_test_probs.append(test_pred["prob"].to_numpy(dtype=float))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        oof_df = pd.concat(model_oof, ignore_index=True).sort_values("file_name").reset_index(drop=True)
        oof_by_model[model_name] = oof_df
        oof_df.to_csv(Path(args.output_dir) / f"oof_{model_name}.csv", index=False)
        test_prob_by_model[model_name] = np.stack(model_test_probs, axis=0).mean(axis=0)

    oof_ensemble, best_config = grid_search_ensemble(oof_by_model, args.models)
    oof_ensemble.to_csv(Path(args.output_dir) / "oof_ensemble.csv", index=False)
    save_json(best_config, Path(args.output_dir) / "best_oof_ensemble_config.json")

    test_probs = test_df[["file_name"]].copy()
    ensemble_prob = np.zeros(len(test_probs), dtype=float)
    for model_name, weight in best_config["best_weights"].items():
        ensemble_prob += test_prob_by_model[model_name] * float(weight)
    test_probs["prob"] = ensemble_prob
    test_probs.to_csv(Path(args.output_dir) / "test_probs.csv", index=False)

    submission = create_submission(
        test_probs,
        threshold=float(best_config["best_threshold"]),
        sample_submission=args.sample_submission,
        submission_dir=args.submission_dir,
    )
    create_threshold_submissions(
        test_probs,
        sample_submission=args.sample_submission,
        submission_dir=args.submission_dir,
    )

    print("Transfer ensemble complete.")
    print(json.dumps(to_jsonable(best_config | {"submission_rows": len(submission)}), indent=2))


if __name__ == "__main__":
    main()
