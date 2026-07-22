"""Train one transfer-learning model on Kaggle.

Project framing:
- Research/education portfolio project only.
- Not a clinical diagnostic system.
- Label mapping: 0 = NORMAL, 1 = PNEUMONIA.

How to use on Kaggle:
1. Attach a Kaggle dataset containing:
   - train_split.csv
   - val_split.csv
   - test.csv
   - sample_submission.csv
   - image directory
2. Edit DATA_ROOT and CONFIG paths below.
3. Change CONFIG["experiment_name"] to one of:
   - densenet121_frozen
   - densenet121_finetune
   - resnet50_frozen
   - resnet50_finetune
   - efficientnet_b0_frozen
   - efficientnet_b0_finetune
   - densenet201
   - efficientnet_v2_s
   - resnext50_32x4d
4. Run this script or paste the full file into a Kaggle notebook cell.

Fine-tuning note:
- Run the matching frozen experiment first when possible.
- Fine-tuning experiments automatically look for:
  /kaggle/working/transfer_learning/<backbone>_frozen/best_model.pth
  unless CONFIG["frozen_checkpoint_path"] is set.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import argparse
from pathlib import Path
from typing import Any

# Local macOS scientific Python stacks can load duplicate OpenMP runtimes through
# torch/sklearn/numpy. This workaround is for local experimentation only.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Keep pretrained torchvision weights inside the project when running locally.
# This avoids permission issues on systems where ~/.cache/torch is restricted.
if not Path("/kaggle/input").exists():
    _local_project_root = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
    os.environ.setdefault("TORCH_HOME", str(_local_project_root / "outputs" / "torch_cache"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 1. CONFIG
# =============================================================================

IS_KAGGLE = Path("/kaggle/input").exists()
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()

if IS_KAGGLE:
    DATA_ROOT = "/kaggle/input/<DATASET_NAME>"
    WORK_DIR = "/kaggle/working"
    DEFAULT_TRAIN_SPLIT_CSV = f"{DATA_ROOT}/train_split.csv"
    DEFAULT_VAL_SPLIT_CSV = f"{DATA_ROOT}/val_split.csv"
    DEFAULT_TEST_CSV = f"{DATA_ROOT}/test.csv"
    DEFAULT_SAMPLE_SUBMISSION_CSV = f"{DATA_ROOT}/sample_submission.csv"
    DEFAULT_IMAGE_DIR = f"{DATA_ROOT}/images"
    DEFAULT_OUTPUT_ROOT = f"{WORK_DIR}/outputs"
else:
    DATA_ROOT = str(PROJECT_ROOT)
    WORK_DIR = str(PROJECT_ROOT / "outputs")
    DEFAULT_TRAIN_SPLIT_CSV = str(PROJECT_ROOT / "outputs" / "train_split.csv")
    DEFAULT_VAL_SPLIT_CSV = str(PROJECT_ROOT / "outputs" / "val_split.csv")
    DEFAULT_TEST_CSV = str(PROJECT_ROOT / "data" / "test.csv")
    DEFAULT_SAMPLE_SUBMISSION_CSV = str(PROJECT_ROOT / "data" / "sample_submission.csv")
    DEFAULT_IMAGE_DIR = str(PROJECT_ROOT / "data" / "images")
    DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "outputs" / "transfer_learning")

EXPERIMENTS = {
    "densenet121_frozen": {"backbone": "densenet121", "regime": "frozen"},
    "densenet121_finetune": {"backbone": "densenet121", "regime": "finetune"},
    "resnet50_frozen": {"backbone": "resnet50", "regime": "frozen"},
    "resnet50_finetune": {"backbone": "resnet50", "regime": "finetune"},
    "efficientnet_b0_frozen": {"backbone": "efficientnet_b0", "regime": "frozen"},
    "efficientnet_b0_finetune": {"backbone": "efficientnet_b0", "regime": "finetune"},
    "densenet201": {
        "backbone": "densenet201",
        "regime": "finetune",
        "image_size": 224,
        "load_frozen_checkpoint": False,
    },
    "efficientnet_v2_s": {
        "backbone": "efficientnet_v2_s",
        "regime": "finetune",
        "image_size": 384,
        "load_frozen_checkpoint": False,
    },
    "resnext50_32x4d": {
        "backbone": "resnext50_32x4d",
        "regime": "finetune",
        "image_size": 224,
        "load_frozen_checkpoint": False,
    },
}

MODEL_NAME = "densenet201"

CONFIG = {
    # MODEL_NAME examples for the additional comparison/diversity experiments:
    # "densenet201", "efficientnet_v2_s", "resnext50_32x4d"
    "experiment_name": MODEL_NAME,
    # Set this to None to run only CONFIG["experiment_name"].
    # To run all additional models sequentially, set this to:
    # ["densenet201", "efficientnet_v2_s", "resnext50_32x4d"]
    "run_experiments": None,
    "train_split_csv": DEFAULT_TRAIN_SPLIT_CSV,
    "val_split_csv": DEFAULT_VAL_SPLIT_CSV,
    "test_csv": DEFAULT_TEST_CSV,
    "sample_submission_csv": DEFAULT_SAMPLE_SUBMISSION_CSV,
    "image_dir": DEFAULT_IMAGE_DIR,
    "output_root": DEFAULT_OUTPUT_ROOT,
    "frozen_checkpoint_path": None,
    "seed": 42,
    "label_mapping": {"0": "NORMAL", "1": "PNEUMONIA"},
    "image_size": 224,
    "image_size_override": None,
    "batch_size": 32,
    "epochs": 10,
    "num_workers": 2,
    "device": "auto",  # "auto", "cuda", "mps", or "cpu"
    "threshold": 0.5,
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "use_pos_weight": True,
    "head_lr": 1e-3,
    "backbone_lr": 1e-5,
    "scheduler": "ReduceLROnPlateau",  # "ReduceLROnPlateau", "CosineAnnealingLR", or None
    "scheduler_factor": 0.5,
    "scheduler_patience": 2,
    "cosine_t_max": 10,
    "early_stopping_patience": 5,
    "smoke_test": False,
    "smoke_train_samples": 64,
    "smoke_val_samples": 32,
    "smoke_test_samples": 32,
    "use_amp": True,
    "efficientnet_unfreeze_blocks": 3,
    "best_submission_metric": "auroc",
    "run_test_inference": True,
    "resume": False,
}


# =============================================================================
# 2. Seed and paths
# =============================================================================


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def prepare_output_dir(config: dict[str, Any]) -> Path:
    output_dir = Path(config["output_root"]) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CONFIG['device']='cuda' but CUDA is not available.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("CONFIG['device']='mps' but Apple MPS is not available.")
    return device


# =============================================================================
# 3. Dataset
# =============================================================================


class ChestXrayCsvDataset(Dataset):
    """CSV-backed image dataset.

    Training and validation CSVs must include file_name and label.
    Test CSV only needs file_name.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform: Any | None = None,
        require_label: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.require_label = require_label
        self.df = pd.read_csv(self.csv_path)

        if "file_name" not in self.df.columns:
            raise ValueError(f"{self.csv_path} must contain a file_name column.")
        if require_label and "label" not in self.df.columns:
            raise ValueError(f"{self.csv_path} must contain a label column.")

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, file_name: str) -> Path:
        raw = Path(str(file_name))
        candidates = [
            self.image_dir / raw,
            self.image_dir / raw.name,
            self.image_dir / "train" / raw.name,
            self.image_dir / "val" / raw.name,
            self.image_dir / "test" / raw.name,
            self.image_dir.parent / raw,
            self.image_dir.parent / "images" / raw,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        tried = ", ".join(str(path) for path in candidates[:5])
        raise FileNotFoundError(f"Could not resolve {file_name}. Tried: {tried}")

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        file_name = str(row["file_name"])
        image = Image.open(self.resolve_path(file_name)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.require_label:
            label = float(row["label"])
            return image, torch.tensor(label, dtype=torch.float32), file_name
        return image, file_name


# =============================================================================
# 4. Transform
# =============================================================================


def require_torchvision():
    try:
        from torchvision import models, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for transfer learning experiments. "
            "Kaggle GPU notebooks usually include it with PyTorch."
        ) from exc
    return models, transforms


def build_transforms(image_size: int) -> tuple[Any, Any]:
    _models, transforms = require_torchvision()
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.96, 1.04)),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            # Vertical flip is intentionally excluded for chest X-rays because it creates
            # anatomically implausible images and can distort clinically meaningful orientation.
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return train_transform, eval_transform


def make_loaders(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, ChestXrayCsvDataset, ChestXrayCsvDataset]:
    train_transform, eval_transform = build_transforms(config["image_size"])
    train_dataset = ChestXrayCsvDataset(config["train_split_csv"], config["image_dir"], train_transform, require_label=True)
    val_dataset = ChestXrayCsvDataset(config["val_split_csv"], config["image_dir"], eval_transform, require_label=True)
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


def sample_smoke_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True).copy()
    if "label" in df.columns:
        sampled_parts = []
        labels = sorted(df["label"].dropna().unique().tolist())
        per_label = max(1, n // max(len(labels), 1))
        for label in labels:
            part = df[df["label"] == label]
            sampled_parts.append(part.sample(min(len(part), per_label), random_state=seed))
        sampled = pd.concat(sampled_parts)
        if len(sampled) < n:
            remaining = df.drop(sampled.index, errors="ignore")
            if len(remaining) > 0:
                sampled = pd.concat(
                    [sampled, remaining.sample(min(len(remaining), n - len(sampled)), random_state=seed)],
                )
        return sampled.sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def apply_smoke_sampling(
    train_dataset: ChestXrayCsvDataset,
    val_dataset: ChestXrayCsvDataset,
    config: dict[str, Any],
) -> None:
    train_dataset.df = sample_smoke_df(train_dataset.df, int(config["smoke_train_samples"]), int(config["seed"]))
    val_dataset.df = sample_smoke_df(val_dataset.df, int(config["smoke_val_samples"]), int(config["seed"]) + 1)


def validate_duplicate_group_split(train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    if "duplicate_group_id" not in train_df.columns or "duplicate_group_id" not in val_df.columns:
        print("duplicate_group_id column not found in both split files; skipping duplicate-group leakage check.")
        return
    train_groups = set(train_df["duplicate_group_id"].dropna().astype(str))
    val_groups = set(val_df["duplicate_group_id"].dropna().astype(str))
    overlap = sorted(train_groups & val_groups)
    if overlap:
        raise ValueError(
            "duplicate_group_id leakage detected between train_split.csv and val_split.csv. "
            f"First overlapping groups: {overlap[:10]}"
        )
    print("Duplicate-group split check passed: no train/validation duplicate_group_id overlap.")


def print_loader_batch_check(loader: DataLoader, name: str, require_label: bool) -> None:
    batch = next(iter(loader))
    if require_label:
        images, labels, file_names = batch
        label_counts = pd.Series(labels.numpy().astype(int)).value_counts().sort_index().to_dict()
        print(f"{name} batch shape: images={tuple(images.shape)} labels={tuple(labels.shape)} label_counts={label_counts}")
    else:
        images, file_names = batch
        print(f"{name} batch shape: images={tuple(images.shape)} rows={len(file_names)}")
    print(f"{name} first files: {list(file_names)[:5]}")


# =============================================================================
# 5. Model builder
# =============================================================================


def build_model(backbone: str, dropout: float) -> nn.Module:
    """Build a pretrained torchvision model with a single binary logit head."""
    models, _transforms = require_torchvision()
    name = backbone.lower()

    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        # DenseNet classifier replacement: output [B, 1] logit for BCEWithLogitsLoss.
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    elif name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        # ResNet classifier replacement: fc becomes a binary logit head.
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    elif name == "resnext50_32x4d":
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        # EfficientNet classifier replacement: classifier emits [B, 1].
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    elif name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


def classifier_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    raise ValueError("Could not find classifier head.")


def freeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def apply_frozen_regime(model: nn.Module) -> None:
    """Frozen regime: freeze the pretrained backbone and train only the new head."""
    freeze_all(model)
    unfreeze_module(classifier_module(model))


def apply_finetune_regime(model: nn.Module, backbone: str, efficientnet_unfreeze_blocks: int = 3) -> None:
    """Fine-tuning regime: unfreeze only high-level backbone blocks plus classifier."""
    freeze_all(model)
    name = backbone.lower()

    if name in {"densenet121", "densenet201"}:
        unfreeze_module(model.features.denseblock4)
        unfreeze_module(model.features.norm5)
        unfreeze_module(model.classifier)
    elif name in {"resnet50", "resnext50_32x4d"}:
        unfreeze_module(model.layer4)
        unfreeze_module(model.fc)
    elif name in {"efficientnet_b0", "efficientnet_v2_s"}:
        blocks_to_unfreeze = max(1, int(efficientnet_unfreeze_blocks))
        for block in model.features[-blocks_to_unfreeze:]:
            unfreeze_module(block)
        unfreeze_module(model.classifier)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


def load_frozen_checkpoint_if_needed(model: nn.Module, config: dict[str, Any], output_dir: Path) -> None:
    if config["regime"] != "finetune":
        return
    if not config.get("load_frozen_checkpoint", True):
        return

    checkpoint_path = config.get("frozen_checkpoint_path")
    if checkpoint_path is None:
        frozen_name = f"{config['backbone']}_frozen"
        checkpoint_path = Path(config["output_root"]) / frozen_name / "best_model.pth"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"No frozen checkpoint found at {checkpoint_path}. Fine-tuning will start from ImageNet weights.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    (output_dir / "loaded_frozen_checkpoint.txt").write_text(str(checkpoint_path), encoding="utf-8")
    print(f"Loaded frozen checkpoint for fine-tuning: {checkpoint_path}")


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    head_params = list(classifier_module(model).parameters())
    head_param_ids = {id(param) for param in head_params}
    backbone_params = [param for param in model.parameters() if param.requires_grad and id(param) not in head_param_ids]

    if config["regime"] == "frozen":
        param_groups = [{"params": head_params, "lr": config["head_lr"]}]
    else:
        param_groups = [
            {"params": backbone_params, "lr": config["backbone_lr"]},
            {"params": head_params, "lr": config["head_lr"]},
        ]
    return torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    scheduler_name = config.get("scheduler")
    if scheduler_name is None:
        return None
    if scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
        )
    if scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["cosine_t_max"])
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# =============================================================================
# 6. Metrics
# =============================================================================


def compute_pos_weight(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor | None:
    counts = train_df["label"].value_counts().sort_index()
    negative = int(counts.get(0, 0))
    positive = int(counts.get(1, 0))
    if positive == 0:
        return None
    return torch.tensor([negative / positive], dtype=torch.float32, device=device)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    # BCEWithLogitsLoss trains on raw logits. Sigmoid(logit) gives P(PNEUMONIA).
    # Prediction uses threshold 0.5 by default: prob >= 0.5 -> PNEUMONIA(1), else NORMAL(0).
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "normal_recall": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "pneumonia_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError as exc:
        metrics["auroc"] = None
        metrics["auroc_reason"] = str(exc)
    return metrics


def is_better_checkpoint(current: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    current_auroc = current.get("auroc")
    best_auroc = best.get("auroc")
    current_score = -math.inf if current_auroc is None else float(current_auroc)
    best_score = -math.inf if best_auroc is None else float(best_auroc)
    if current_score > best_score:
        return True
    if current_score == best_score and float(current["f1"]) > float(best["f1"]):
        return True
    return False


# =============================================================================
# 7. Train and validate
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    amp_enabled = bool(use_amp and device.type == "cuda")

    for images, labels, _file_names in loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool = False,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    records: list[dict[str, Any]] = []
    amp_enabled = bool(use_amp and device.type == "cuda")

    with torch.no_grad():
        for images, labels, file_names in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

            for i in range(batch_size):
                true_label = int(labels[i].detach().cpu().item())
                prob_pneumonia = float(probs[i].detach().cpu().item())
                pred_label = int(prob_pneumonia >= threshold)
                records.append(
                    {
                        "file_name": file_names[i],
                        "image_id": file_names[i],
                        "true_label": true_label,
                        "true_class": "NORMAL" if true_label == 0 else "PNEUMONIA",
                        "logit": float(logits[i].detach().cpu().item()),
                        "pred_prob": prob_pneumonia,
                        "probability_PNEUMONIA": prob_pneumonia,
                        "probability_NORMAL": 1.0 - prob_pneumonia,
                        "pred_label": pred_label,
                        "pred_class": "NORMAL" if pred_label == 0 else "PNEUMONIA",
                        "is_correct": bool(pred_label == true_label),
                    }
                )

    pred_df = pd.DataFrame(records)
    metrics = compute_metrics(
        pred_df["true_label"].to_numpy(dtype=int),
        pred_df["probability_PNEUMONIA"].to_numpy(dtype=float),
        threshold,
    )
    return total_loss / max(total_items, 1), metrics, pred_df


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "config": to_jsonable(config),
        "metrics": to_jsonable(metrics),
        "label_mapping": {0: "NORMAL", 1: "PNEUMONIA"},
    }
    torch.save(checkpoint, path)


def save_validation_outputs(metrics: dict[str, Any], pred_df: pd.DataFrame, output_dir: Path) -> None:
    cm_df = pd.DataFrame(
        np.asarray(metrics["confusion_matrix"], dtype=int),
        index=["actual_NORMAL_0", "actual_PNEUMONIA_1"],
        columns=["pred_NORMAL_0", "pred_PNEUMONIA_1"],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    false_negatives = pred_df[(pred_df["true_label"] == 1) & (pred_df["pred_label"] == 0)].copy()
    false_positives = pred_df[(pred_df["true_label"] == 0) & (pred_df["pred_label"] == 1)].copy()
    false_negatives.to_csv(output_dir / "false_negatives.csv", index=False)
    false_positives.to_csv(output_dir / "false_positives.csv", index=False)
    pred_df.to_csv(output_dir / "val_predictions.csv", index=False)
    pred_df.to_csv(output_dir / "validation_predictions.csv", index=False)


def load_resume_state(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None, pd.DataFrame | None, int | None]:
    checkpoint_path = output_dir / "latest_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Cannot resume because latest checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    completed_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = completed_epoch + 1

    logs: list[dict[str, Any]] = []
    train_log_path = output_dir / "train_log.csv"
    if train_log_path.exists():
        logs = pd.read_csv(train_log_path).to_dict(orient="records")

    best_metrics = None
    best_epoch = None
    best_metrics_path = output_dir / "best_metrics.json"
    if best_metrics_path.exists():
        best_metrics = json.loads(best_metrics_path.read_text(encoding="utf-8"))
        if best_metrics.get("epoch") is not None:
            best_epoch = int(best_metrics["epoch"])

    best_pred_df = None
    best_pred_path = output_dir / "validation_predictions.csv"
    if best_pred_path.exists():
        best_pred_df = pd.read_csv(best_pred_path)

    print(f"Resuming from {checkpoint_path}; completed epoch {completed_epoch}, next epoch {start_epoch}.")
    return start_epoch, logs, best_metrics, best_pred_df, best_epoch


# =============================================================================
# 8. Inference and submission
# =============================================================================


def inference(model: nn.Module, config: dict[str, Any], device: torch.device) -> pd.DataFrame:
    _train_transform, eval_transform = build_transforms(config["image_size"])
    test_dataset = ChestXrayCsvDataset(config["test_csv"], config["image_dir"], eval_transform, require_label=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model.eval()
    records: list[dict[str, Any]] = []
    amp_enabled = bool(config.get("use_amp", True) and device.type == "cuda")
    with torch.no_grad():
        for images, file_names in test_loader:
            images = images.to(device)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
            probs = torch.sigmoid(logits).view(-1)
            for file_name, logit, prob in zip(file_names, logits.view(-1), probs):
                probability = float(prob.detach().cpu().item())
                records.append(
                    {
                        "file_name": file_name,
                        "image_id": file_name,
                        "logit": float(logit.detach().cpu().item()),
                        "pred_prob": probability,
                        "probability_PNEUMONIA": probability,
                        "label": int(probability >= config["threshold"]),
                    }
                )
    return pd.DataFrame(records)


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame, target_col: str) -> None:
    if list(submission.columns) != list(sample.columns):
        raise ValueError(
            "submission.csv columns must exactly match sample_submission.csv columns. "
            f"Expected {list(sample.columns)}, got {list(submission.columns)}"
        )
    if len(submission) != len(sample):
        raise ValueError(f"submission.csv row count mismatch: expected {len(sample)}, got {len(submission)}")
    if not submission["file_name"].equals(sample["file_name"]):
        raise ValueError("submission.csv file_name order must match sample_submission.csv.")
    values = submission[target_col].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("submission.csv contains NaN prediction values.")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("submission.csv prediction values must be probabilities in [0, 1].")


def create_submission(pred_df: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    sample = pd.read_csv(config["sample_submission_csv"])
    if "file_name" not in sample.columns:
        raise ValueError("sample_submission.csv must contain file_name column.")

    target_cols = [col for col in sample.columns if col != "file_name"]
    if len(target_cols) != 1:
        raise ValueError(f"Expected one target column in sample_submission.csv, found: {target_cols}")
    target_col = target_cols[0]

    merged = sample[["file_name"]].merge(pred_df, on="file_name", how="left")
    if merged["probability_PNEUMONIA"].isna().any():
        missing = merged.loc[merged["probability_PNEUMONIA"].isna(), "file_name"].head().tolist()
        raise ValueError(f"Missing predictions for sample rows: {missing}")

    submission = sample[["file_name"]].copy()
    # Keep sample_submission.csv structure exactly, but submit P(PNEUMONIA).
    submission[target_col] = merged["probability_PNEUMONIA"].astype(float)
    validate_submission(submission, sample, target_col)

    submission.to_csv(output_dir / "submission.csv", index=False)
    return submission


# =============================================================================
# 9. Experiment runner
# =============================================================================


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    experiment_name = resolved["experiment_name"]
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment_name: {experiment_name}. Choose from {sorted(EXPERIMENTS)}")
    resolved.update(EXPERIMENTS[experiment_name])
    if resolved.get("image_size_override") is not None:
        resolved["image_size"] = int(resolved["image_size_override"])
    return resolved


def run_experiment() -> dict[str, Any]:
    config = resolve_config(CONFIG)
    return run_single_experiment(config)


def run_single_experiment(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    if config.get("smoke_test", False):
        config["epochs"] = 1
        config["num_workers"] = 0
        config["run_test_inference"] = True

    set_seed(config["seed"])
    output_dir = prepare_output_dir(config)
    save_json(config, output_dir / "config.json")

    device = get_device(config.get("device", "auto"))
    print(f"Experiment: {config['experiment_name']}")
    print(f"Backbone: {config['backbone']} | Regime: {config['regime']}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if device.type == "mps":
        print("MPS device: Apple Metal Performance Shaders")

    full_train_df = pd.read_csv(config["train_split_csv"])
    full_val_df = pd.read_csv(config["val_split_csv"])
    validate_duplicate_group_split(full_train_df, full_val_df)

    train_loader, val_loader, train_dataset, val_dataset = make_loaders(config, device)
    if config.get("smoke_test", False):
        apply_smoke_sampling(train_dataset, val_dataset, config)

    train_df = train_dataset.df.copy()
    val_df = val_dataset.df.copy()
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print("Train class counts:")
    print(train_df["label"].value_counts().sort_index())
    print("Validation class counts:")
    print(val_df["label"].value_counts().sort_index())
    if config.get("smoke_test", False):
        _train_transform, eval_transform = build_transforms(config["image_size"])
        test_dataset = ChestXrayCsvDataset(config["test_csv"], config["image_dir"], eval_transform, require_label=False)
        test_dataset.df = test_dataset.df.head(int(config["smoke_test_samples"])).reset_index(drop=True)
        smoke_test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        print_loader_batch_check(train_loader, "Smoke train", require_label=True)
        print_loader_batch_check(val_loader, "Smoke validation", require_label=True)
        print_loader_batch_check(smoke_test_loader, "Smoke test", require_label=False)

    model = build_model(config["backbone"], config["dropout"])
    load_frozen_checkpoint_if_needed(model, config, output_dir)
    if config["regime"] == "frozen":
        apply_frozen_regime(model)
    else:
        apply_finetune_regime(model, config["backbone"], config["efficientnet_unfreeze_blocks"])
    model = model.to(device)

    trainable_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters. Check freeze/unfreeze settings.")

    pos_weight = compute_pos_weight(train_df, device) if config["use_pos_weight"] else None
    print(f"pos_weight for BCEWithLogitsLoss: {None if pos_weight is None else pos_weight.detach().cpu().tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler("cuda", enabled=bool(config.get("use_amp", True) and device.type == "cuda"))

    logs: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_pred_df: pd.DataFrame | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    start_epoch = 1

    if config.get("resume", False):
        start_epoch, logs, best_metrics, best_pred_df, best_epoch = load_resume_state(
            output_dir,
            model,
            optimizer,
            scheduler,
            device,
        )
        if best_metrics is not None and logs:
            best_log_index = max(0, int(best_epoch or len(logs)) - 1)
            epochs_without_improvement = max(0, len(logs) - best_log_index - 1)

    for epoch in range(start_epoch, config["epochs"] + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            use_amp=bool(config.get("use_amp", True)),
        )
        val_loss, val_metrics, pred_df = validate(
            model,
            val_loader,
            criterion,
            device,
            config["threshold"],
            use_amp=bool(config.get("use_amp", True)),
        )

        if scheduler is not None:
            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(-math.inf if val_metrics["auroc"] is None else val_metrics["auroc"])
            else:
                scheduler.step()

        lr_values = [group["lr"] for group in optimizer.param_groups]
        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "auroc": val_metrics.get("auroc"),
            "normal_recall": val_metrics["normal_recall"],
            "pneumonia_recall": val_metrics["pneumonia_recall"],
            "sensitivity": val_metrics["sensitivity"],
            "specificity": val_metrics["specificity"],
            "fn": val_metrics["fn"],
            "fp": val_metrics["fp"],
            "lr_group_0": lr_values[0] if len(lr_values) > 0 else None,
            "lr_group_1": lr_values[1] if len(lr_values) > 1 else None,
            "epoch_seconds": time.time() - start,
        }
        logs.append(log_row)
        pd.DataFrame(logs).to_csv(output_dir / "train_log.csv", index=False)

        epoch_metrics = val_metrics | {"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch}
        print(
            f"Epoch {epoch:02d}/{config['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"auroc={val_metrics.get('auroc')} f1={val_metrics['f1']:.4f} "
            f"pneu_recall={val_metrics['pneumonia_recall']:.4f} normal_recall={val_metrics['normal_recall']:.4f} "
            f"fn={val_metrics['fn']} fp={val_metrics['fp']}"
        )

        save_checkpoint(output_dir / "latest_model.pth", model, optimizer, scheduler, epoch, config, epoch_metrics)

        if is_better_checkpoint(epoch_metrics, best_metrics):
            best_metrics = epoch_metrics
            best_pred_df = pred_df.copy()
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best_model.pth", model, optimizer, scheduler, epoch, config, epoch_metrics)
            save_json(epoch_metrics, output_dir / "best_metrics.json")
            save_validation_outputs(epoch_metrics, best_pred_df, output_dir)
        else:
            epochs_without_improvement += 1

        patience = config.get("early_stopping_patience")
        if patience is not None and epochs_without_improvement >= int(patience):
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_metrics is None or best_pred_df is None or best_epoch is None:
        raise RuntimeError("Training ended before a best checkpoint could be selected.")

    final_metrics = logs[-1] | {"best_epoch": best_epoch, "best_auroc": best_metrics.get("auroc"), "best_f1": best_metrics["f1"]}
    save_json(final_metrics, output_dir / "final_metrics.json")

    submission_rows = None
    if config.get("run_test_inference", True):
        best_checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_pred_df = inference(model, config, device)
        test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
        submission = create_submission(test_pred_df, config, output_dir)
        submission_rows = len(submission)
        print(
            "Test prediction range: "
            f"min={test_pred_df['pred_prob'].min():.6f}, "
            f"max={test_pred_df['pred_prob'].max():.6f}, "
            f"rows={len(test_pred_df)}"
        )
        print(f"submission.csv generated and validated: rows={submission_rows}, columns={list(submission.columns)}")

    results = {
        "experiment_name": config["experiment_name"],
        "backbone": config["backbone"],
        "regime": config["regime"],
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "trainable_parameters": trainable_params,
        "output_dir": str(output_dir),
        "submission_rows": submission_rows,
        "run_test_inference": bool(config.get("run_test_inference", True)),
    }
    save_json(results, output_dir / "run_summary.json")
    return results


def summarize_run_result(result: dict[str, Any]) -> dict[str, Any]:
    best_metrics = result["best_metrics"]
    final_metrics = result["final_metrics"]
    return {
        "experiment_name": result["experiment_name"],
        "backbone": result["backbone"],
        "regime": result["regime"],
        "best_epoch": result["best_epoch"],
        "best_auroc": best_metrics.get("auroc"),
        "best_f1": best_metrics.get("f1"),
        "best_accuracy": best_metrics.get("accuracy"),
        "best_precision": best_metrics.get("precision"),
        "best_recall": best_metrics.get("recall"),
        "best_pneumonia_recall": best_metrics.get("pneumonia_recall"),
        "best_normal_recall": best_metrics.get("normal_recall"),
        "best_sensitivity": best_metrics.get("sensitivity"),
        "best_specificity": best_metrics.get("specificity"),
        "best_fn": best_metrics.get("fn"),
        "best_fp": best_metrics.get("fp"),
        "final_auroc": final_metrics.get("auroc"),
        "final_f1": final_metrics.get("f1"),
        "trainable_parameters": result["trainable_parameters"],
        "output_dir": result["output_dir"],
        "submission_path": str(Path(result["output_dir"]) / "submission.csv") if result.get("run_test_inference") else None,
    }


def select_best_submission(summary_df: pd.DataFrame, metric: str) -> pd.Series:
    metric_col = f"best_{metric}"
    if metric_col not in summary_df.columns:
        raise ValueError(f"Cannot select best submission. Missing column: {metric_col}")

    sortable = summary_df.copy()
    sortable[metric_col] = pd.to_numeric(sortable[metric_col], errors="coerce")
    sortable["best_f1"] = pd.to_numeric(sortable["best_f1"], errors="coerce")
    sortable = sortable.sort_values(
        by=[metric_col, "best_f1", "best_pneumonia_recall"],
        ascending=[False, False, False],
        na_position="last",
    )
    return sortable.iloc[0]


def run_configured_experiments() -> dict[str, Any]:
    run_names = CONFIG.get("run_experiments")
    if run_names is None:
        run_names = [CONFIG["experiment_name"]]

    unknown = [name for name in run_names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown run_experiments entries: {unknown}. Choose from {sorted(EXPERIMENTS)}")

    output_root = Path(CONFIG["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for experiment_name in run_names:
        experiment_config = dict(CONFIG)
        experiment_config["experiment_name"] = experiment_name
        experiment_config = resolve_config(experiment_config)

        print("=" * 80)
        print(f"Starting experiment: {experiment_name}")
        print("=" * 80)
        result = run_single_experiment(experiment_config)
        results.append(result)
        summary_rows.append(summarize_run_result(result))

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_root / "comparison_summary_partial.csv", index=False)
        save_json({"completed": summary_rows}, output_root / "comparison_summary_partial.json")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "comparison_summary.csv", index=False)
    save_json({"results": summary_rows}, output_root / "comparison_summary.json")

    best_row = select_best_submission(summary_df, CONFIG["best_submission_metric"])
    best_selection = {
        "selection_metric": CONFIG["best_submission_metric"],
        "selected_experiment": best_row["experiment_name"],
        "selected_metrics": best_row.to_dict(),
        "note": "This is selected by validation metrics, not by hidden leaderboard performance.",
    }

    final_submission_path = None
    if pd.notna(best_row.get("submission_path")):
        best_submission_path = Path(best_row["submission_path"])
        final_submission_path = output_root / "submission_best_validation_model.csv"
        pd.read_csv(best_submission_path).to_csv(final_submission_path, index=False)
        best_selection["selected_submission_path"] = str(best_submission_path)
        best_selection["copied_submission_path"] = str(final_submission_path)

    save_json(best_selection, output_root / "best_submission_selection.json")

    return {
        "completed_experiments": [row["experiment_name"] for row in summary_rows],
        "comparison_summary": str(output_root / "comparison_summary.csv"),
        "best_submission": str(final_submission_path) if final_submission_path is not None else None,
        "best_submission_selection": best_selection,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transfer learning chest X-ray experiments.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), help="Run one experiment only.")
    parser.add_argument("--model-name", choices=sorted(EXPERIMENTS), help="Alias for --experiment.")
    parser.add_argument(
        "--split",
        choices=["default", "strict"],
        default=None,
        help="Use default split files or strict split files.",
    )
    parser.add_argument("--epochs", type=int, help="Override CONFIG['epochs'].")
    parser.add_argument("--image-size", type=int, help="Override the model-specific default image size.")
    parser.add_argument("--batch-size", type=int, help="Override CONFIG['batch_size'].")
    parser.add_argument("--num-workers", type=int, help="Override CONFIG['num_workers'].")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="Override CONFIG['device'].")
    parser.add_argument("--smoke-test", action="store_true", help="Run a 1-epoch smoke test with small train/val samples.")
    parser.add_argument(
        "--skip-test-inference",
        action="store_true",
        help="Skip test prediction and submission CSV generation. Keeps validation metrics/checkpoints/logs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an experiment from its latest_model.pth checkpoint.",
    )
    return parser.parse_args()


def apply_cli_args(args: argparse.Namespace) -> None:
    selected_experiment = args.model_name or args.experiment
    if selected_experiment:
        CONFIG["experiment_name"] = selected_experiment
        CONFIG["run_experiments"] = [selected_experiment]

    if args.split == "strict":
        CONFIG["train_split_csv"] = str(PROJECT_ROOT / "outputs" / "train_split_strict.csv")
        CONFIG["val_split_csv"] = str(PROJECT_ROOT / "outputs" / "val_split_strict.csv")
        CONFIG["output_root"] = str(PROJECT_ROOT / "outputs" / "transfer_learning_strict")
    elif args.split == "default":
        CONFIG["train_split_csv"] = str(PROJECT_ROOT / "outputs" / "train_split.csv")
        CONFIG["val_split_csv"] = str(PROJECT_ROOT / "outputs" / "val_split.csv")
        CONFIG["output_root"] = str(PROJECT_ROOT / "outputs" / "transfer_learning")

    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
    if args.image_size is not None:
        CONFIG["image_size_override"] = args.image_size
    if args.batch_size is not None:
        CONFIG["batch_size"] = args.batch_size
    if args.num_workers is not None:
        CONFIG["num_workers"] = args.num_workers
    if args.device is not None:
        CONFIG["device"] = args.device
    if args.skip_test_inference:
        CONFIG["run_test_inference"] = False
    if args.resume:
        CONFIG["resume"] = True
    if args.smoke_test:
        CONFIG["smoke_test"] = True


# =============================================================================
# 10. Execute
# =============================================================================


if __name__ == "__main__":
    apply_cli_args(parse_args())
    run_results = run_configured_experiments()
    print("Transfer learning run complete.")
    print(json.dumps(to_jsonable(run_results), indent=2))
