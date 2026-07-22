"""Progressively fine-tune the 384px ensemble at 512px in Colab.

Research/education portfolio project only. This script is not a clinical
diagnostic system and must not be presented as one.

Expected source checkpoint names by default:
- best_densenet121_384_fold{0..4}.pt
- best_convnext_384_fold{0..4}.pt
- best_efficientnet_384_fold{0..4}.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from google.colab import drive
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset


drive.mount("/content/drive")


class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smoothed_targets)


# ==========================================
# 1. Colab paths and Phase 3 defaults
# ==========================================
PROJECT_ROOT = "/content/drive/MyDrive/xray_project"
DATA_DIR = f"{PROJECT_ROOT}/data"
OUTPUTS_DIR = f"{PROJECT_ROOT}/outputs"
PHASE1_OUTPUT_DIR = OUTPUTS_DIR

SOURCE_IMAGE_SIZE = 384
IMAGE_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 2
LR_BACKBONE = 1e-6
LR_HEAD = 1e-5

MODEL_ALIASES = {
    "densenet121": "densenet121",
    "convnext": "convnext",
    "convnext_tiny": "convnext",
    "efficientnet": "efficientnet",
    "efficientnet_b3": "efficientnet",
}
MODEL_FILE_STEMS = {
    "densenet121": "densenet121",
    "convnext": "convnext",
    "efficientnet": "efficientnet",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TTA_MODES = ("crop", "crop_flip", "full", "full_flip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 high-resolution fine-tune 384px ensemble checkpoints.")
    parser.add_argument("--train-csv", default=f"{DATA_DIR}/train_split_strict.csv")
    parser.add_argument("--val-csv", default=f"{DATA_DIR}/val_split_strict.csv")
    parser.add_argument("--test-csv", default=f"{DATA_DIR}/test.csv")
    parser.add_argument("--sample-submission", default=f"{DATA_DIR}/sample_submission.csv")
    parser.add_argument("--image-dir", default=f"{DATA_DIR}/images")
    parser.add_argument("--checkpoint-dir", default=PHASE1_OUTPUT_DIR)
    parser.add_argument("--output-dir", default=f"{OUTPUTS_DIR}/phase3_highres_finetune_512")
    parser.add_argument("--submission-dir", default=f"{OUTPUTS_DIR}/phase3_highres_finetune_512/submissions")
    parser.add_argument("--models", nargs="+", default=["densenet121", "convnext", "efficientnet"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--source-image-size", type=int, default=SOURCE_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr-backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--lr-head", type=float, default=LR_HEAD)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pseudo-high-threshold", type=float, default=0.95)
    parser.add_argument("--pseudo-low-threshold", type=float, default=0.05)
    parser.add_argument("--pseudo-loss-weight", type=float, default=0.35)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--max-pseudo-samples", type=int, default=None)
    parser.add_argument("--use-val-in-train", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Only validate checkpoint/model compatibility.")
    parser.add_argument("--debug", action="store_true", help="Fast smoke mode: fold 0, one epoch, small pseudo subset.")
    args, _unknown = parser.parse_known_args()
    return args


def normalize_models(models: list[str]) -> list[str]:
    normalized = []
    for model in models:
        key = model.lower()
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unsupported model: {model}. Choose from {sorted(MODEL_ALIASES)}")
        canonical = MODEL_ALIASES[key]
        if canonical not in normalized:
            normalized.append(canonical)
    return normalized


def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.models = normalize_models(args.models)
    if args.debug:
        args.folds = [0]
        args.epochs = 1
        args.num_workers = 0
        args.max_pseudo_samples = 32 if args.max_pseudo_samples is None else args.max_pseudo_samples
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.submission_dir).mkdir(parents=True, exist_ok=True)
    return args


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


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def require_torchvision():
    try:
        from torchvision import models, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torchvision is required. In Colab, run: pip install torchvision") from exc
    return models, transforms


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class CBAM(nn.Module):
    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DenseNet121CBAM(nn.Module):
    def __init__(self, base: nn.Module, dropout: float) -> None:
        super().__init__()
        in_features = base.classifier.in_features
        self.features = base.features
        self.cbam = CBAM(in_features)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = torch.relu(features)
        features = self.cbam(features)
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        return self.classifier(flattened)


def build_model(model_name: str, dropout: float) -> nn.Module:
    models, _transforms = require_torchvision()
    if model_name == "densenet121":
        base = models.densenet121(weights=None)
        return DenseNet121CBAM(base, dropout=dropout)
    if model_name == "convnext":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model
    if model_name == "efficientnet":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def checkpoint_path(checkpoint_dir: str | Path, model_name: str, fold: int, source_image_size: int) -> Path:
    stem = MODEL_FILE_STEMS[model_name]
    stem_candidates = [stem]
    if model_name == "convnext":
        stem_candidates.append("convnext_tiny")
    if model_name == "efficientnet":
        stem_candidates.append("efficientnet_b3")

    candidates = []
    for candidate_stem in stem_candidates:
        candidates.extend(
            [
                Path(checkpoint_dir) / f"best_{candidate_stem}_{source_image_size}_fold{fold}.pt",
                Path(checkpoint_dir) / "phase1_outputs" / f"best_{candidate_stem}_{source_image_size}_fold{fold}.pt",
                Path(checkpoint_dir) / "Densenet+efficientnet_B3+convnext_384" / f"best_{candidate_stem}_{source_image_size}_fold{fold}.pt",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type at {path}: {type(checkpoint)}")


def load_model(model_name: str, path: str | Path, dropout: float, device: torch.device) -> nn.Module:
    model = build_model(model_name, dropout=dropout)
    state_dict = load_state_dict(path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {path}: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return model.to(device)


def build_transform(image_size: int, train: bool = False, tta_mode: str = "full") -> Any:
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
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if tta_mode.startswith("crop"):
        ops: list[Any] = [transforms.Resize(int(round(image_size * 1.14))), transforms.CenterCrop(image_size)]
    elif tta_mode.startswith("full"):
        ops = [transforms.Resize((image_size, image_size))]
    else:
        raise ValueError(f"Unknown tta_mode: {tta_mode}")
    if tta_mode.endswith("flip"):
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        transform: Any | None,
        require_label: bool = True,
        sample_weight: float = 1.0,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.require_label = require_label
        self.sample_weight = float(sample_weight)
        if "file_name" not in self.df.columns:
            raise ValueError("DataFrame must contain file_name column.")
        if require_label and "label" not in self.df.columns:
            raise ValueError("Labeled DataFrame must contain label column.")

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

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        file_name = str(row["file_name"])
        image_path = self.resolve_path(file_name)
        if image_path is None:
            print(f"WARNING: image file not found for {file_name}; using fallback black image.")
            image = Image.new("RGB", (512, 512), color=(0, 0, 0))
        else:
            image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if not self.require_label:
            return image, file_name
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        weight = torch.tensor(float(row.get("sample_weight", self.sample_weight)), dtype=torch.float32)
        return image, label, weight, file_name


def make_eval_loader(df: pd.DataFrame, args: argparse.Namespace, tta_mode: str, require_label: bool, batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        ChestXrayDataset(df, args.image_dir, build_transform(args.image_size, train=False, tta_mode=tta_mode), require_label=require_label),
        batch_size=batch_size or args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device, require_label: bool) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        if require_label:
            images, labels, _weights, file_names = batch
        else:
            images, file_names = batch
            labels = None
        images = images.to(device)
        probs = torch.sigmoid(model(images)).view(-1).detach().cpu().numpy()
        for idx, file_name in enumerate(file_names):
            row = {"file_name": file_name, "prob": float(probs[idx])}
            if labels is not None:
                row["label"] = int(labels[idx].item())
            rows.append(row)
    return pd.DataFrame(rows)


def predict_tta(model: nn.Module, df: pd.DataFrame, args: argparse.Namespace, device: torch.device, require_label: bool) -> pd.DataFrame:
    modes = ("full",) if args.no_tta else TTA_MODES
    mode_dfs = []
    for mode in modes:
        loader = make_eval_loader(df, args, mode, require_label=require_label)
        mode_dfs.append(predict_model(model, loader, device, require_label=require_label))
    merged = mode_dfs[0][["file_name"]].copy()
    if require_label:
        merged["label"] = mode_dfs[0]["label"].astype(int)
    merged["prob"] = np.stack([mode_df["prob"].to_numpy(dtype=float) for mode_df in mode_dfs], axis=0).mean(axis=0)
    return merged


def validate_checkpoints(args: argparse.Namespace, device: torch.device) -> None:
    dummy = torch.zeros(2, 3, args.image_size, args.image_size, device=device)
    rows = []
    for model_name in args.models:
        for fold in args.folds:
            path = checkpoint_path(args.checkpoint_dir, model_name, fold, args.source_image_size)
            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {path}")
            model = load_model(model_name, path, args.dropout, device)
            model.eval()
            with torch.no_grad():
                output = model(dummy)
            if tuple(output.shape) != (2, 1):
                raise RuntimeError(f"Unexpected output shape for {path}: {tuple(output.shape)}")
            rows.append({"model": model_name, "fold": fold, "checkpoint": str(path), "output_shape": list(output.shape)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(Path(args.output_dir) / "checkpoint_compatibility.csv", index=False)
    print(f"Checkpoint compatibility OK: {len(rows)} checkpoints")


def generate_pseudo_labels(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    test_df = pd.read_csv(args.test_csv)
    prob_columns = []
    for model_name in args.models:
        for fold in args.folds:
            path = checkpoint_path(args.checkpoint_dir, model_name, fold, args.source_image_size)
            model = load_model(model_name, path, args.dropout, device)
            pred = predict_tta(model, test_df, args, device, require_label=False)
            col = f"prob_{model_name}_fold{fold}"
            test_df[col] = pred["prob"].to_numpy(dtype=float)
            prob_columns.append(col)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    test_df["prob"] = test_df[prob_columns].mean(axis=1)
    test_df["label"] = (test_df["prob"] >= 0.5).astype(int)
    confident = test_df[(test_df["prob"] <= args.pseudo_low_threshold) | (test_df["prob"] >= args.pseudo_high_threshold)].copy()
    confident["pseudo_confidence"] = np.maximum(confident["prob"], 1.0 - confident["prob"])
    confident["sample_weight"] = args.pseudo_loss_weight
    if args.max_pseudo_samples is not None and len(confident) > args.max_pseudo_samples:
        confident = confident.sort_values("pseudo_confidence", ascending=False).head(args.max_pseudo_samples).copy()

    output_dir = Path(args.output_dir)
    test_df.to_csv(output_dir / "test_ensemble_probs_before_phase3.csv", index=False)
    confident[["file_name", "label", "prob", "pseudo_confidence", "sample_weight"]].to_csv(output_dir / "phase3_pseudo_labels.csv", index=False)
    return confident[["file_name", "label", "sample_weight"]].copy()


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": auroc,
        "pneumonia_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    criterion = LabelSmoothingBCEWithLogitsLoss(smoothing=0.1)
    for images, labels, weights, _file_names in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)
        weights = weights.to(device).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_weight = float(weights.sum().item())
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight
    return total_loss / max(total_weight, 1.0)


def make_train_loader(train_df: pd.DataFrame, pseudo_df: pd.DataFrame, args: argparse.Namespace) -> DataLoader:
    train_df = train_df.copy()
    train_df["sample_weight"] = 1.0
    datasets: list[Dataset] = [
        ChestXrayDataset(train_df, args.image_dir, build_transform(args.image_size, train=True), require_label=True)
    ]
    if not pseudo_df.empty:
        datasets.append(
            ChestXrayDataset(pseudo_df, args.image_dir, build_transform(args.image_size, train=True), require_label=True)
        )
    combined = ConcatDataset(datasets)
    return DataLoader(
        combined,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def optimizer_for_model(model: nn.Module, model_name: str, args: argparse.Namespace) -> torch.optim.Optimizer:
    if model_name == "densenet121":
        head_params = list(model.classifier.parameters()) + list(model.cbam.parameters())
        head_ids = {id(param) for param in head_params}
        backbone_params = [param for param in model.parameters() if id(param) not in head_ids]
    elif model_name in {"convnext", "efficientnet"}:
        head_params = list(model.classifier.parameters())
        head_ids = {id(param) for param in head_params}
        backbone_params = [param for param in model.parameters() if id(param) not in head_ids]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )


def fine_tune(args: argparse.Namespace, pseudo_df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    if args.use_val_in_train:
        train_df = pd.concat([train_df, val_df], ignore_index=True)

    rows = []
    output_dir = Path(args.output_dir)
    ft_dir = output_dir / "finetuned_checkpoints"
    ft_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        for fold in args.folds:
            source_path = checkpoint_path(args.checkpoint_dir, model_name, fold, args.source_image_size)
            model = load_model(model_name, source_path, args.dropout, device)
            optimizer = optimizer_for_model(model, model_name, args)
            train_loader = make_train_loader(train_df, pseudo_df, args)

            best: dict[str, Any] | None = None
            best_path = ft_dir / f"phase3_{model_name}_{args.image_size}_fold{fold}.pt"
            for epoch in range(1, args.epochs + 1):
                start = time.time()
                train_loss = train_one_epoch(model, train_loader, optimizer, device)
                val_pred = predict_tta(model, val_df, args, device, require_label=True)
                metrics = binary_metrics(
                    val_pred["label"].to_numpy(dtype=int),
                    val_pred["prob"].to_numpy(dtype=float),
                    args.decision_threshold,
                )
                row = {
                    "model": model_name,
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "source_checkpoint": str(source_path),
                    "lr_backbone": args.lr_backbone,
                    "lr_head": args.lr_head,
                    **metrics,
                    "seconds": time.time() - start,
                }
                rows.append(row)
                print(
                    f"{model_name} fold={fold} epoch={epoch}/{args.epochs} "
                    f"loss={train_loss:.4f} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
                    f"auroc={metrics['auroc']} pneu_recall={metrics['pneumonia_recall']:.4f} "
                    f"normal_recall={metrics['normal_recall']:.4f} fn={metrics['fn']} fp={metrics['fp']}"
                )
                if best is None or metrics["accuracy"] > best["accuracy"]:
                    best = row
                    torch.save(model.state_dict(), best_path)

            save_json(best or {}, output_dir / f"best_metrics_phase3_{model_name}_fold{fold}.json")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    log_df = pd.DataFrame(rows)
    log_df.to_csv(output_dir / "phase3_highres_finetune_log.csv", index=False)
    return log_df


def final_inference(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    test_df = pd.read_csv(args.test_csv)
    prob_cols = []
    for model_name in args.models:
        for fold in args.folds:
            path = Path(args.output_dir) / "finetuned_checkpoints" / f"phase3_{model_name}_{args.image_size}_fold{fold}.pt"
            model = load_model(model_name, path, args.dropout, device)
            pred = predict_tta(model, test_df, args, device, require_label=False)
            col = f"prob_phase3_{model_name}_fold{fold}"
            test_df[col] = pred["prob"].to_numpy(dtype=float)
            prob_cols.append(col)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    test_df["prob"] = test_df[prob_cols].mean(axis=1)
    test_df[["file_name", "prob"]].to_csv(Path(args.output_dir) / "test_probs_phase3_highres_512.csv", index=False)
    return test_df[["file_name", "prob"]].copy()


def create_submission(test_probs: pd.DataFrame, args: argparse.Namespace) -> None:
    sample = pd.read_csv(args.sample_submission)
    target_cols = [col for col in sample.columns if col != "file_name"]
    if len(target_cols) != 1:
        raise ValueError(f"Expected one target column in sample submission, got {target_cols}")
    target_col = target_cols[0]
    merged = sample[["file_name"]].merge(test_probs, on="file_name", how="left")
    if merged["prob"].isna().any():
        missing = merged.loc[merged["prob"].isna(), "file_name"].head().tolist()
        raise ValueError(f"Missing probabilities for sample rows: {missing}")
    submission = sample[["file_name"]].copy()
    submission[target_col] = (merged["prob"].to_numpy(dtype=float) >= args.decision_threshold).astype(int)
    submission.to_csv(Path(args.submission_dir) / "submission_phase3_highres_512.csv", index=False)
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        threshold_submission = sample[["file_name"]].copy()
        threshold_submission[target_col] = (merged["prob"].to_numpy(dtype=float) >= threshold).astype(int)
        threshold_submission.to_csv(Path(args.submission_dir) / f"submission_phase3_highres_512_threshold_{threshold:.2f}.csv", index=False)


def require_input_paths(args: argparse.Namespace) -> None:
    for path, label in [
        (args.train_csv, "train CSV"),
        (args.val_csv, "validation CSV"),
        (args.test_csv, "test CSV"),
        (args.sample_submission, "sample submission"),
        (args.image_dir, "image directory"),
        (args.checkpoint_dir, "source checkpoint directory"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> None:
    args = prepare_args(parse_args())
    set_seed(args.seed)
    require_input_paths(args)
    save_json(vars(args), Path(args.output_dir) / "config_phase3_highres_512.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"DATA_DIR={DATA_DIR}")
    print(f"OUTPUTS_DIR={OUTPUTS_DIR}")
    print(f"PHASE1_OUTPUT_DIR={args.checkpoint_dir}")
    print(f"Models: {args.models}; folds: {args.folds}")
    print(f"Source image size={args.source_image_size}; train image size={args.image_size}")
    print(f"Batch size={args.batch_size}; epochs={args.epochs}")
    print(f"LR_BACKBONE={args.lr_backbone}; LR_HEAD={args.lr_head}")
    print("Criterion: LabelSmoothingBCEWithLogitsLoss(smoothing=0.1)")

    validate_checkpoints(args, device)
    if args.check_only:
        print("Check-only mode complete.")
        return

    pseudo_df = generate_pseudo_labels(args, device)
    print(f"Pseudo labels selected: {len(pseudo_df)}")
    fine_tune(args, pseudo_df, device)
    test_probs = final_inference(args, device)
    create_submission(test_probs, args)
    print("Phase 3 progressive resizing fine-tuning complete.")


if __name__ == "__main__":
    main()
