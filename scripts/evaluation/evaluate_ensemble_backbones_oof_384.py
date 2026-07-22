"""Evaluate 384px ensemble backbones with existing OOF fold checkpoints.

Research/education portfolio project only. The metrics produced here are for
model comparison and should not be described as clinical diagnostic validity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


MODEL_NAMES = ("convnext", "efficientnet")
MODEL_ALIASES = {
    "convnext": "convnext",
    "convnext_tiny": "convnext",
    "efficientnet": "efficientnet",
    "efficientnet_b3": "efficientnet",
}
DISPLAY_NAMES = {
    "convnext": "ConvNeXt-Tiny 384",
    "efficientnet": "EfficientNet-B3 384",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing 384px fold checkpoints as single models.")
    parser.add_argument("--train-csv", default="outputs/train_split.csv")
    parser.add_argument("--val-csv", default="outputs/val_split.csv")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--checkpoint-dir", default="outputs/Densenet+efficientnet_B3+convnext_384")
    parser.add_argument("--output-dir", default="outputs/single_model_384_eval")
    parser.add_argument("--models", nargs="+", default=["convnext", "efficientnet"])
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tta", choices=["crop", "four_way"], default="crop")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-val-samples-per-fold", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        img_np = np.array(img_pil)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        cl = self.clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str | Path, transform: Any) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, file_name: str) -> Path:
        raw = Path(str(file_name))
        candidates = [
            self.image_dir / raw,
            self.image_dir / "train" / raw.name,
            self.image_dir / "test" / raw.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image not found for file_name={file_name}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        file_name = str(row["file_name"])
        image = Image.open(self.resolve_path(file_name)).convert("RGB")
        image = self.transform(image)
        return image, int(row["label"]), file_name


def build_transforms(image_size: int) -> tuple[Any, Any]:
    crop_transform = transforms.Compose(
        [
            CLAHETransform(),
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    full_transform = transforms.Compose(
        [
            CLAHETransform(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return crop_transform, full_transform


def canonical_model_name(model_name: str) -> str:
    key = model_name.lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {sorted(MODEL_ALIASES)}")
    return MODEL_ALIASES[key]


def build_model(model_name: str) -> nn.Module:
    if model_name == "convnext":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 1)
        return model
    if model_name == "efficientnet":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def checkpoint_path(checkpoint_dir: Path, model_name: str, fold: int) -> Path:
    path = checkpoint_dir / f"best_{model_name}_384_fold{fold}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def split_folds(full_df: pd.DataFrame, num_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if "duplicate_group_id" not in full_df.columns:
        raise ValueError("Expected duplicate_group_id column for strict group-aware fold recreation.")
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return list(splitter.split(full_df, full_df["label"], groups=full_df["duplicate_group_id"]))


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    probs: list[float] = []
    labels: list[int] = []
    file_names: list[str] = []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels, batch_names in loader:
            images = images.to(device)
            batch_probs = torch.sigmoid(model(images)).detach().cpu().numpy().reshape(-1)
            probs.extend(float(prob) for prob in batch_probs)
            labels.extend(int(label) for label in batch_labels.numpy().reshape(-1))
            file_names.extend(str(name) for name in batch_names)
    return np.array(probs, dtype=float), np.array(labels, dtype=int), file_names


def predict_four_way(
    model: nn.Module,
    crop_loader: DataLoader,
    full_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    probs: list[float] = []
    labels: list[int] = []
    file_names: list[str] = []
    model.eval()
    with torch.inference_mode():
        for crop_batch, full_batch in zip(crop_loader, full_loader):
            crop_images, batch_labels, batch_names = crop_batch
            full_images, _, _ = full_batch
            crop_images = crop_images.to(device)
            full_images = full_images.to(device)
            batch_probs = (
                torch.sigmoid(model(crop_images))
                + torch.sigmoid(model(torch.flip(crop_images, dims=[3])))
                + torch.sigmoid(model(full_images))
                + torch.sigmoid(model(torch.flip(full_images, dims=[3])))
            ) / 4.0
            probs.extend(float(prob) for prob in batch_probs.detach().cpu().numpy().reshape(-1))
            labels.extend(int(label) for label in batch_labels.numpy().reshape(-1))
            file_names.extend(str(name) for name in batch_names)
    return np.array(probs, dtype=float), np.array(labels, dtype=int), file_names


def calculate_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "pneumonia_recall": recall_score(labels, preds, zero_division=0),
        "normal_recall": recall_score(labels, preds, pos_label=0, zero_division=0),
        "sensitivity": recall_score(labels, preds, zero_division=0),
        "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auroc": auroc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in np.round(np.arange(0.1, 0.9001, 0.02), 2):
        metrics = calculate_metrics(labels, probs, float(threshold))
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    if best is None:
        raise RuntimeError("Threshold search failed.")
    return best


def evaluate_model(model_name: str, full_df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    crop_transform, full_transform = build_transforms(args.image_size)
    oof_rows: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold, (_, val_idx) in enumerate(folds):
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        if args.max_val_samples_per_fold is not None:
            val_df = val_df.head(args.max_val_samples_per_fold).copy()
        print(f"  Fold {fold}/{len(folds) - 1}: {len(val_df)} validation samples", flush=True)
        model = build_model(model_name).to(device)
        state_dict = torch.load(checkpoint_path(checkpoint_dir, model_name, fold), map_location=device)
        model.load_state_dict(state_dict)

        crop_loader = DataLoader(
            ChestXrayDataset(val_df, args.image_dir, crop_transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        if args.tta == "four_way":
            full_loader = DataLoader(
                ChestXrayDataset(val_df, args.image_dir, full_transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            probs, labels, file_names = predict_four_way(model, crop_loader, full_loader, device)
        else:
            probs, labels, file_names = predict_loader(model, crop_loader, device)

        fold_result = calculate_metrics(labels, probs, args.threshold)
        fold_result.update({"model": model_name, "fold": fold})
        fold_metrics.append(fold_result)

        oof_rows.append(
            pd.DataFrame(
                {
                    "file_name": file_names,
                    "label": labels,
                    "fold": fold,
                    "prob": probs,
                }
            )
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof_df = pd.concat(oof_rows, ignore_index=True).sort_values("file_name").reset_index(drop=True)
    labels = oof_df["label"].to_numpy(dtype=int)
    probs = oof_df["prob"].to_numpy(dtype=float)
    fixed_metrics = calculate_metrics(labels, probs, args.threshold)
    tuned_metrics = best_threshold_metrics(labels, probs)

    preds = (probs > args.threshold).astype(int)
    result_df = oof_df.copy()
    result_df["pred"] = preds
    result_df.to_csv(output_dir / f"oof_{model_name}.csv", index=False)
    result_df[(result_df["label"] == 1) & (result_df["pred"] == 0)].to_csv(output_dir / f"false_negatives_{model_name}.csv", index=False)
    result_df[(result_df["label"] == 0) & (result_df["pred"] == 1)].to_csv(output_dir / f"false_positives_{model_name}.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir / f"fold_metrics_{model_name}.csv", index=False)
    pd.DataFrame(
        confusion_matrix(labels, preds, labels=[0, 1]),
        index=["actual_normal", "actual_pneumonia"],
        columns=["pred_normal", "pred_pneumonia"],
    ).to_csv(output_dir / f"confusion_matrix_{model_name}.csv")

    summary = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "tta": args.tta,
        "fixed_threshold_metrics": fixed_metrics,
        "best_f1_threshold_metrics": tuned_metrics,
    }
    save_json(summary, output_dir / f"metrics_{model_name}.json")
    print(
        f"  Done {DISPLAY_NAMES[model_name]}: "
        f"F1={fixed_metrics['f1']:.4f}, AUROC={fixed_metrics['auroc']:.4f}, "
        f"FN={fixed_metrics['fn']}, FP={fixed_metrics['fp']}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    args.models = [canonical_model_name(model_name) for model_name in args.models]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_json(vars(args), Path(args.output_dir) / "config.json")

    set_seed(args.seed)
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    folds = split_folds(full_df, args.num_folds, args.seed)

    summaries = []
    for model_name in args.models:
        print(f"Evaluating {DISPLAY_NAMES[model_name]} on {device}...")
        summaries.append(evaluate_model(model_name, full_df, folds, args, device))

    rows = []
    for summary in summaries:
        metrics = summary["fixed_threshold_metrics"]
        tuned = summary["best_f1_threshold_metrics"]
        rows.append(
            {
                "model": summary["model"],
                "display_name": summary["display_name"],
                "tta": summary["tta"],
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "sensitivity_pneumonia_recall": metrics["sensitivity"],
                "specificity_normal_recall": metrics["specificity"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "fn": metrics["fn"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "tp": metrics["tp"],
                "best_f1_threshold": tuned["threshold"],
                "best_f1": tuned["f1"],
                "best_f1_fn": tuned["fn"],
                "best_f1_fp": tuned["fp"],
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    summary_df.to_csv(Path(args.output_dir) / "summary.csv", index=False)
    save_json({"results": summaries}, Path(args.output_dir) / "summary.json")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
