"""Train the strict-split OOF-calibrated 384px ensemble on Kaggle.

Default experiment: strict_focal_no_clahe_letterbox_384_nofold

Binary labels:
- NORMAL = 0
- PNEUMONIA = 1

This script is intended to be pasted into or run from a Kaggle Notebook. It
preserves the existing backbone family while exposing ablation flags for
letterbox padding, CLAHE, focal loss, MixUp, optional CBAM wrappers, OOF/validation
calibration, and submission sanity checks.
"""

from __future__ import annotations

import gc
import json
import os
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ==========================================
# 1. Configuration & Paths
# ==========================================
DATA_ROOT = "/kaggle/input/datasets/hyunwoo11/chest-xray-ai"
WORK_DIR = "/kaggle/working"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/datasets/hyunwoo11/submission/sample_submission.csv"

TRAIN_CSV = f"{DATA_ROOT}/train_split_strict.csv"
VAL_CSV = f"{DATA_ROOT}/val_split_strict.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
IMAGE_DIR = f"{DATA_ROOT}/data/images"

DEBUG_MODE = False
NO_FOLD_MODE = True
USE_LETTERBOX = True
USE_CLAHE = False
USE_FOCAL_LOSS = True
USE_MIXUP = False
ADD_CBAM_TO_ALL = False
ENSEMBLE_OPT_METRIC = "accuracy"

FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 1.5

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2

# Full run uses max 15 epochs with accuracy-based early stopping to reduce
# Kaggle runtime risk. This is a practical trade-off; 20 epochs may be tested
# later if runtime is sufficient.
if DEBUG_MODE:
    EPOCHS = 2
    NUM_FOLDS = 2
    IMAGE_SIZE = 224
    MODEL_NAMES = ["densenet121"]
    DEBUG_TRAIN_SAMPLES_PER_FOLD = 256
elif NO_FOLD_MODE:
    EPOCHS = 20
    NUM_FOLDS = 1
    IMAGE_SIZE = 384
    MODEL_NAMES = ["densenet121", "convnext_tiny", "efficientnet_b3"]
    DEBUG_TRAIN_SAMPLES_PER_FOLD = None
else:
    EPOCHS = 15
    NUM_FOLDS = 5
    IMAGE_SIZE = 384
    MODEL_NAMES = ["densenet121", "convnext_tiny", "efficientnet_b3"]
    DEBUG_TRAIN_SAMPLES_PER_FOLD = None

BATCH_SIZE = 16
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.1
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0005
EARLY_STOPPING_START_EPOCH = 5
TOP_K_CHECKPOINTS = 3

if ENSEMBLE_OPT_METRIC != "accuracy":
    raise ValueError("This Kaggle leaderboard is scored by accuracy; ENSEMBLE_OPT_METRIC must remain 'accuracy'.")

LOSS_NAME = "focal" if USE_FOCAL_LOSS else "bce"
CLAHE_NAME = "clahe" if USE_CLAHE else "no_clahe"
PREPROCESS_NAME = "letterbox" if USE_LETTERBOX else "crop"
MODE_NAME = "nofold" if NO_FOLD_MODE else "5fold"
EXPERIMENT_NAME = f"strict_{LOSS_NAME}_{CLAHE_NAME}_{PREPROCESS_NAME}_{IMAGE_SIZE}_{MODE_NAME}"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_DISPLAY_NAMES = {
    "densenet121": "DenseNet121",
    "convnext_tiny": "ConvNeXt-Tiny",
    "efficientnet_b3": "EfficientNet-B3",
}
CHECKPOINT_NAMES = {
    "densenet121": "densenet121",
    "convnext_tiny": "convnext_tiny",
    "efficientnet_b3": "efficientnet_b3",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)


# ==========================================
# 2. Preprocessing & Dataset
# ==========================================
class CLAHETransform:
    """Optional contrast enhancement. Default is off for safer ablation."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        img_np = np.array(img_pil)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        cl = self.clahe.apply(l_channel)
        limg = cv2.merge((cl, a_channel, b_channel))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)


class PadToSquare:
    def __init__(self, fill: tuple[int, int, int] = (0, 0, 0)):
        self.fill = fill

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        image = img_pil.convert("RGB")
        width, height = image.size
        side = max(width, height)
        padded = Image.new("RGB", (side, side), self.fill)
        left = (side - width) // 2
        top = (side - height) // 2
        padded.paste(image, (left, top))
        return padded


class KaggleChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, file_name: str) -> str:
        paths_to_check = [
            os.path.join(self.image_dir, file_name),
            os.path.join(self.image_dir, "train", file_name),
            os.path.join(self.image_dir, "test", file_name),
        ]
        for path in paths_to_check:
            if os.path.exists(path):
                return path
        return paths_to_check[0]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_name = row["file_name"]
        img_path = self.resolve_path(file_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {file_name}. Checked under {self.image_dir}")

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.is_test:
            return image, file_name
        return image, int(row["label"])


def build_transforms() -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    train_steps = []
    eval_crop_steps = []
    eval_full_steps = []

    if USE_CLAHE:
        # CLAHE is applied before geometric resizing/padding so contrast is enhanced
        # on the original pixel distribution rather than on padded black borders.
        train_steps.append(CLAHETransform())
        eval_crop_steps.append(CLAHETransform())
        eval_full_steps.append(CLAHETransform())

    if USE_LETTERBOX:
        train_steps.extend([PadToSquare(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))])
        eval_crop_steps.extend([PadToSquare(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))])
        eval_full_steps.extend([PadToSquare(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))])
    else:
        train_steps.extend(
            [
                transforms.Resize((int(IMAGE_SIZE * 1.05), int(IMAGE_SIZE * 1.05))),
                transforms.CenterCrop(IMAGE_SIZE),
            ]
        )
        eval_crop_steps.extend(
            [
                transforms.Resize((int(IMAGE_SIZE * 1.05), int(IMAGE_SIZE * 1.05))),
                transforms.CenterCrop(IMAGE_SIZE),
            ]
        )
        eval_full_steps.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))

    train_steps.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tail = [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    eval_crop_steps.extend(eval_tail)
    eval_full_steps.extend(eval_tail)
    return transforms.Compose(train_steps), transforms.Compose(eval_crop_steps), transforms.Compose(eval_full_steps)


train_transform, val_test_transform_crop, val_test_transform_full = build_transforms()


# ==========================================
# 3. Model Architectures
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class DenseNet121CBAM(nn.Module):
    def __init__(self, weights):
        super().__init__()
        base = models.densenet121(weights=weights)
        self.features = base.features
        self.cbam = CBAM(in_planes=1024)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = self.cbam(features)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


class ConvNeXtTinyCBAM(nn.Module):
    def __init__(self, weights):
        super().__init__()
        base = models.convnext_tiny(weights=weights)
        self.features = base.features
        self.cbam = CBAM(in_planes=768)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cbam(self.features(x))
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


class EfficientNetB3CBAM(nn.Module):
    def __init__(self, weights):
        super().__init__()
        base = models.efficientnet_b3(weights=weights)
        self.features = base.features
        self.cbam = CBAM(in_planes=1536)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1536, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cbam(self.features(x))
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_factor * (1.0 - pt).pow(self.gamma)
        return (focal_weight * bce).mean()


def create_criterion() -> nn.Module:
    if USE_FOCAL_LOSS:
        print(f"Using BinaryFocalLoss(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
        return BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    print("Using BCEWithLogitsLoss")
    return nn.BCEWithLogitsLoss()


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    mixed_labels = lam * labels + (1.0 - lam) * labels[index]
    return mixed_images, mixed_labels


def resolve_pretrained_weights(model_name: str):
    weights_by_model = {
        "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "efficientnet_b3": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
    }
    return weights_by_model[model_name]


def handle_pretrained_load_failure(model_name: str, exc: Exception):
    if DEBUG_MODE:
        warnings.warn(
            f"Failed to load ImageNet pretrained weights for {MODEL_DISPLAY_NAMES[model_name]} in DEBUG_MODE. "
            f"Falling back to weights=None. Error: {exc}",
            RuntimeWarning,
        )
        return None
    raise RuntimeError(
        f"Failed to load required ImageNet pretrained weights for {MODEL_DISPLAY_NAMES[model_name]}. "
        "Full run does not allow silent weights=None fallback."
    ) from exc


def create_model(model_name: str) -> nn.Module:
    weights = resolve_pretrained_weights(model_name)
    print(f"Loading ImageNet pretrained weights for {MODEL_DISPLAY_NAMES[model_name]}: {weights}")
    try:
        if model_name == "densenet121":
            model = DenseNet121CBAM(weights=weights)
        elif model_name == "convnext_tiny":
            if ADD_CBAM_TO_ALL:
                model = ConvNeXtTinyCBAM(weights=weights)
            else:
                model = models.convnext_tiny(weights=weights)
                in_features = model.classifier[2].in_features
                model.classifier[2] = nn.Linear(in_features, 1)
        elif model_name == "efficientnet_b3":
            if ADD_CBAM_TO_ALL:
                model = EfficientNetB3CBAM(weights=weights)
            else:
                model = models.efficientnet_b3(weights=weights)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        print(f"ImageNet pretrained weights loaded for {MODEL_DISPLAY_NAMES[model_name]}")
    except Exception as exc:
        weights = handle_pretrained_load_failure(model_name, exc)
        if model_name == "densenet121":
            model = DenseNet121CBAM(weights=weights)
        elif model_name == "convnext_tiny":
            if ADD_CBAM_TO_ALL:
                model = ConvNeXtTinyCBAM(weights=weights)
            else:
                model = models.convnext_tiny(weights=weights)
                in_features = model.classifier[2].in_features
                model.classifier[2] = nn.Linear(in_features, 1)
        elif model_name == "efficientnet_b3":
            if ADD_CBAM_TO_ALL:
                model = EfficientNetB3CBAM(weights=weights)
            else:
                model = models.efficientnet_b3(weights=weights)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, 1)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
    model = model.to(DEVICE)
    if DEBUG_MODE and ADD_CBAM_TO_ALL and model_name in {"convnext_tiny", "efficientnet_b3"}:
        model.eval()
        with torch.no_grad():
            dummy_out = model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE))
        if tuple(dummy_out.shape) != (2, 1):
            raise RuntimeError(f"{model_name} CBAM wrapper output shape must be [2, 1], got {tuple(dummy_out.shape)}")
        print(f"{MODEL_DISPLAY_NAMES[model_name]} CBAM wrapper dummy output shape verified: {tuple(dummy_out.shape)}")
    return model


def get_param_groups(model_name: str, model: nn.Module):
    if model_name == "densenet121":
        backbone_params = model.features.parameters()
        head_params = list(model.classifier.parameters()) + list(model.cbam.parameters())
    elif model_name == "convnext_tiny":
        backbone_params = model.features.parameters()
        if ADD_CBAM_TO_ALL:
            head_params = list(model.classifier.parameters()) + list(model.cbam.parameters())
        else:
            head_params = model.classifier.parameters()
    elif model_name == "efficientnet_b3":
        backbone_params = model.features.parameters()
        if ADD_CBAM_TO_ALL:
            head_params = list(model.classifier.parameters()) + list(model.cbam.parameters())
        else:
            head_params = model.classifier.parameters()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return backbone_params, head_params


# ==========================================
# 4. Metrics, Loading, and Checkpoint Utilities
# ==========================================
def require_file(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required {description} not found: {path}\n"
            "Attach the Kaggle input dataset that contains train_split_strict.csv, "
            "val_split_strict.csv, test.csv, images, and sample_submission.csv."
        )


def safe_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    normal_recall = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    pneumonia_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "auroc": safe_auroc(labels, probs),
        "pneumonia_recall": float(pneumonia_recall),
        "normal_recall": float(normal_recall),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_accuracy_threshold(
    labels: np.ndarray, probs: np.ndarray, threshold_min: float = 0.20, threshold_max: float = 0.80, step: float = 0.001
) -> tuple[float, dict[str, float | int]]:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    best_threshold = 0.5
    best_accuracy = -1.0
    for threshold in np.arange(threshold_min, threshold_max + (step / 2), step):
        preds = (probs >= threshold).astype(int)
        accuracy = accuracy_score(labels, preds)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_threshold = round(float(threshold), 3)
    return best_threshold, binary_metrics(labels, probs, threshold=best_threshold)


def average_checkpoints(checkpoint_paths: list[str]) -> dict[str, torch.Tensor]:
    state_dicts = [torch.load(path, map_location=DEVICE) for path in checkpoint_paths]
    averaged_state_dict = {}
    for key in state_dicts[0].keys():
        values = [state_dict[key] for state_dict in state_dicts]
        if torch.is_tensor(values[0]) and values[0].is_floating_point():
            averaged_state_dict[key] = torch.stack(values).mean(dim=0)
        else:
            averaged_state_dict[key] = values[0]
    return averaged_state_dict


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def select_group_column(full_df: pd.DataFrame) -> str:
    if "strict_duplicate_group_id" in full_df.columns:
        return "strict_duplicate_group_id"
    if "duplicate_group_id" in full_df.columns:
        warnings.warn(
            "strict_duplicate_group_id is missing. Falling back to duplicate_group_id.",
            RuntimeWarning,
        )
        return "duplicate_group_id"
    raise KeyError("Neither strict_duplicate_group_id nor duplicate_group_id exists in the split CSV files.")


def log_split_info(train_df: pd.DataFrame, val_df: pd.DataFrame, full_df: pd.DataFrame, group_col: str) -> None:
    train_groups = set(train_df[group_col].astype(str))
    val_groups = set(val_df[group_col].astype(str))
    overlap = sorted(train_groups.intersection(val_groups))
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Device: {DEVICE}")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    print(f"TRAIN_CSV: {TRAIN_CSV}")
    print(f"VAL_CSV: {VAL_CSV}")
    print(f"train rows: {len(train_df)}")
    print(f"val rows: {len(val_df)}")
    print(f"full rows: {len(full_df)}")
    print("label distribution:")
    print(full_df["label"].value_counts().sort_index().to_string())
    print(f"selected group column: {group_col}")
    print(f"group overlap between provided train/val split: {len(overlap) > 0} (count={len(overlap)})")


def log_run_config() -> None:
    print("\nRun configuration")
    print(f"DEBUG_MODE: {DEBUG_MODE}")
    print(f"NO_FOLD_MODE: {NO_FOLD_MODE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"NUM_FOLDS: {NUM_FOLDS}")
    print(f"IMAGE_SIZE: {IMAGE_SIZE}")
    print(f"MODEL_NAMES: {MODEL_NAMES}")
    print(f"EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
    print(f"EARLY_STOPPING_START_EPOCH: {EARLY_STOPPING_START_EPOCH}")
    print(f"TOP_K_CHECKPOINTS: {TOP_K_CHECKPOINTS}")
    print(f"USE_LETTERBOX: {USE_LETTERBOX}")
    print(f"USE_CLAHE: {USE_CLAHE}")
    print(f"USE_FOCAL_LOSS: {USE_FOCAL_LOSS}")
    print(f"USE_MIXUP: {USE_MIXUP}")
    print(f"MIXUP_ALPHA: {MIXUP_ALPHA}")
    print(f"ADD_CBAM_TO_ALL: {ADD_CBAM_TO_ALL}")
    print(f"ENSEMBLE_OPT_METRIC: {ENSEMBLE_OPT_METRIC}")
    print(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}")
    print("leaderboard metric = accuracy")


def log_fold_distribution(
    full_df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], group_col: str
) -> pd.DataFrame:
    rows = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        train_fold_df = full_df.iloc[train_idx].copy()
        val_fold_df = full_df.iloc[val_idx].copy()
        val_counts = val_fold_df["label"].value_counts().to_dict()
        normal_count = int(val_counts.get(0, 0))
        pneumonia_count = int(val_counts.get(1, 0))
        val_size = int(len(val_fold_df))
        val_groups = val_fold_df[group_col].astype(str)
        train_groups = set(train_fold_df[group_col].astype(str))
        val_group_set = set(val_groups)
        overlap_count = len(train_groups.intersection(val_group_set))
        max_group_size = int(val_groups.value_counts().max()) if val_size > 0 else 0
        row = {
            "fold": fold,
            "val_size": val_size,
            "normal_count": normal_count,
            "pneumonia_count": pneumonia_count,
            "pneumonia_ratio": float(pneumonia_count / val_size) if val_size > 0 else float("nan"),
            "unique_group_count": int(val_groups.nunique()),
            "max_group_size": max_group_size,
            "train_val_group_overlap": bool(overlap_count > 0),
            "train_val_group_overlap_count": int(overlap_count),
            "group_column": group_col,
        }
        rows.append(row)
        print(
            f"Generated fold {fold}: val_size={row['val_size']} "
            f"NORMAL={row['normal_count']} PNEUMONIA={row['pneumonia_count']} "
            f"PNEUMONIA_ratio={row['pneumonia_ratio']:.4f} "
            f"unique_groups={row['unique_group_count']} max_group_size={row['max_group_size']} "
            f"train/val {group_col} overlap={row['train_val_group_overlap']} "
            f"(count={row['train_val_group_overlap_count']})"
        )
    fold_distribution_df = pd.DataFrame(rows)
    out_path = os.path.join(WORK_DIR, "fold_distribution.csv")
    fold_distribution_df.to_csv(out_path, index=False)
    print(f"Saved generated fold distribution: {out_path}")
    return fold_distribution_df


def maybe_debug_sample(df: pd.DataFrame, fold: int) -> pd.DataFrame:
    if DEBUG_TRAIN_SAMPLES_PER_FOLD is None or len(df) <= DEBUG_TRAIN_SAMPLES_PER_FOLD:
        return df
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), DEBUG_TRAIN_SAMPLES_PER_FOLD // 2), random_state=SEED + fold))
        .sample(frac=1.0, random_state=SEED + fold)
        .reset_index(drop=True)
    )


# ==========================================
# 5. Training
# ==========================================
def evaluate_loader_single_tta(model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs_list, labels_list = [], []
    running_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels_float = labels.to(DEVICE).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels_float)
            probs = torch.sigmoid(outputs).detach().cpu().numpy().reshape(-1)
            probs_list.extend(probs)
            labels_list.extend(labels.numpy())
            running_loss += float(loss.item()) * images.size(0)
            sample_count += images.size(0)
    return np.asarray(labels_list).astype(int), np.asarray(probs_list).astype(float), running_loss / max(1, sample_count)


def train_fold(
    model_name: str, fold: int, train_loader: DataLoader, val_loader: DataLoader, output_suffix: str = ""
) -> tuple[str, dict]:
    display_name = MODEL_DISPLAY_NAMES[model_name]
    print(f"\n>>> [Fold {fold + 1}/{NUM_FOLDS}] Training {display_name}")

    model = create_model(model_name)
    criterion = create_criterion()

    backbone_params, head_params = get_param_groups(model_name, model)
    for param in backbone_params:
        param.requires_grad = False
    optimizer = optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = None

    top_checkpoints: list[tuple[float, str]] = []
    ckpt_base = CHECKPOINT_NAMES[model_name]
    best_model_path = os.path.join(WORK_DIR, f"best_{ckpt_base}_{IMAGE_SIZE}_fold{fold}.pt")
    best_single_checkpoint_path = os.path.join(WORK_DIR, f"best_single_{ckpt_base}_{IMAGE_SIZE}_fold{fold}.pt")
    best_val_accuracy = -float("inf")
    best_val_threshold = 0.5
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_epoch = EPOCHS
    early_stopped = False
    epoch_history: list[dict] = []

    for epoch in range(EPOCHS):
        if epoch == 2:
            print(">>> Unfreezing backbone with warmup LR")
            backbone_params, head_params = get_param_groups(model_name, model)
            for param in backbone_params:
                param.requires_grad = True
            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": LR_BACKBONE * 0.1},
                    {"params": head_params, "lr": LR_HEAD},
                ],
                weight_decay=WEIGHT_DECAY,
            )
        elif epoch == 3:
            print(">>> Switching to full backbone LR with cosine annealing")
            backbone_params, head_params = get_param_groups(model_name, model)
            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": LR_BACKBONE},
                    {"params": head_params, "lr": LR_HEAD},
                ],
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - 3), eta_min=1e-6)

        model.train()
        running_loss = 0.0
        num_batches = 0
        for images, labels in tqdm(train_loader, desc=f"{display_name} fold {fold} epoch {epoch + 1}", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()

            if USE_MIXUP:
                images, labels = mixup_batch(images, labels, MIXUP_ALPHA)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        val_labels, val_probs, val_loss = evaluate_loader_single_tta(model, val_loader, criterion)
        best_epoch_threshold, epoch_metrics = find_best_accuracy_threshold(val_labels, val_probs)
        mean_loss = running_loss / max(1, num_batches)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | loss={mean_loss:.4f} "
            f"best_threshold={best_epoch_threshold:.3f} "
            f"val_best_accuracy_over_threshold={epoch_metrics['accuracy']:.4f} "
            f"F1={epoch_metrics['f1']:.4f} "
            f"AUROC={epoch_metrics['auroc']:.4f} "
            f"PNEU_rec={epoch_metrics['pneumonia_recall']:.4f} NORMAL_rec={epoch_metrics['normal_recall']:.4f} "
            f"TN={epoch_metrics['tn']} FP={epoch_metrics['fp']} FN={epoch_metrics['fn']} TP={epoch_metrics['tp']}"
        )
        epoch_history.append(
            {
                "experiment": EXPERIMENT_NAME,
                "mode": MODE_NAME,
                "model": model_name,
                "fold": fold,
                "epoch": epoch + 1,
                "train_loss": mean_loss,
                "val_loss": val_loss,
                "accuracy": epoch_metrics["accuracy"],
                "balanced_accuracy": epoch_metrics["balanced_accuracy"],
                "f1": epoch_metrics["f1"],
                "auroc": epoch_metrics["auroc"],
                "pneumonia_recall": epoch_metrics["pneumonia_recall"],
                "normal_recall": epoch_metrics["normal_recall"],
                "best_threshold": best_epoch_threshold,
                "tn": epoch_metrics["tn"],
                "fp": epoch_metrics["fp"],
                "fn": epoch_metrics["fn"],
                "tp": epoch_metrics["tp"],
                "early_stopped": False,
                "checkpoint_path": best_model_path,
            }
        )

        epoch_ckpt = os.path.join(WORK_DIR, f"temp_{ckpt_base}_{IMAGE_SIZE}_fold{fold}_epoch{epoch}.pt")
        torch.save(model.state_dict(), epoch_ckpt)
        current_val_accuracy = float(epoch_metrics["accuracy"])
        top_checkpoints.append((current_val_accuracy, epoch_ckpt))
        top_checkpoints = sorted(top_checkpoints, key=lambda item: item[0], reverse=True)[:TOP_K_CHECKPOINTS]
        keep_paths = {path for _, path in top_checkpoints}
        for path in Path(WORK_DIR).glob(f"temp_{ckpt_base}_{IMAGE_SIZE}_fold{fold}_epoch*.pt"):
            if str(path) not in keep_paths:
                path.unlink(missing_ok=True)

        if current_val_accuracy > best_val_accuracy + EARLY_STOPPING_MIN_DELTA:
            previous_best = best_val_accuracy
            best_val_accuracy = current_val_accuracy
            best_val_threshold = best_epoch_threshold
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_single_checkpoint_path)
            if previous_best == -float("inf"):
                print(f"Epoch {epoch + 1}: val_accuracy improved from -inf to {best_val_accuracy:.4f}. Saving checkpoint.")
            else:
                print(
                    f"Epoch {epoch + 1}: val_accuracy improved from {previous_best:.4f} "
                    f"to {best_val_accuracy:.4f}. Saving checkpoint."
                )
        else:
            epochs_without_improvement += 1
            print(
                f"Epoch {epoch + 1}: val_accuracy did not improve beyond min_delta={EARLY_STOPPING_MIN_DELTA}. "
                f"No improvement for {epochs_without_improvement} epoch(s)."
            )

        if epoch >= EARLY_STOPPING_START_EPOCH and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stopped_epoch = epoch + 1
            early_stopped = True
            epoch_history[-1]["early_stopped"] = True
            print(
                f"Epoch {epoch + 1}: no improvement for {epochs_without_improvement} epochs. "
                "Early stopping triggered."
            )
            break

    print(f"Averaging top-{len(top_checkpoints)} checkpoints for {display_name} fold {fold}")
    averaged_state_dict = average_checkpoints([path for _, path in top_checkpoints])
    torch.save(averaged_state_dict, best_model_path)
    history_path = os.path.join(WORK_DIR, f"epoch_history_{model_name}_fold{fold}{output_suffix}.csv")
    pd.DataFrame(epoch_history).to_csv(history_path, index=False)
    print(f"Saved epoch history: {history_path}")

    for _, path in top_checkpoints:
        Path(path).unlink(missing_ok=True)

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    train_metadata = {
        "stopped_epoch": int(stopped_epoch),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_threshold": float(best_val_threshold),
        "early_stopped": bool(early_stopped),
        "checkpoint_path": best_model_path,
        "epoch_history_path": history_path,
        "epoch_history": epoch_history,
    }
    return best_model_path, train_metadata


# ==========================================
# 6. OOF and Final TTA Inference
# ==========================================
def predict_with_tta(model: nn.Module, loader_crop: DataLoader, loader_full: DataLoader) -> np.ndarray:
    """2-way letterbox TTA or 4-way crop/full TTA depending on preprocessing mode."""
    model.eval()
    probs_list = []
    with torch.no_grad():
        if USE_LETTERBOX:
            for images, _ in loader_crop:
                images = images.to(DEVICE)
                p_orig = torch.sigmoid(model(images))
                p_flip = torch.sigmoid(model(torch.flip(images, dims=[3])))
                batch_probs = (p_orig + p_flip) / 2.0
                probs_list.extend(batch_probs.detach().cpu().numpy().reshape(-1))
        else:
            for (images_crop, _), (images_full, _) in zip(loader_crop, loader_full):
                images_crop = images_crop.to(DEVICE)
                images_full = images_full.to(DEVICE)

                p_crop_orig = torch.sigmoid(model(images_crop))
                p_crop_flip = torch.sigmoid(model(torch.flip(images_crop, dims=[3])))
                p_full_orig = torch.sigmoid(model(images_full))
                p_full_flip = torch.sigmoid(model(torch.flip(images_full, dims=[3])))

                batch_probs = (p_crop_orig + p_crop_flip + p_full_orig + p_full_flip) / 4.0
                probs_list.extend(batch_probs.detach().cpu().numpy().reshape(-1))
    return np.asarray(probs_list, dtype=float)


def make_eval_loaders(df: pd.DataFrame, is_test: bool = False) -> tuple[DataLoader, DataLoader]:
    ds_crop = KaggleChestXrayDataset(df, IMAGE_DIR, transform=val_test_transform_crop, is_test=is_test)
    ds_full = KaggleChestXrayDataset(df, IMAGE_DIR, transform=val_test_transform_full, is_test=is_test)
    loader_crop = DataLoader(ds_crop, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    loader_full = DataLoader(ds_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return loader_crop, loader_full


def build_oof_predictions(
    model_paths: dict[str, list[str]],
    train_metadata: dict[str, list[dict]],
    full_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    output_suffix: str = "",
) -> tuple[dict[str, np.ndarray], list[dict]]:
    oof_probs_by_model = {model_name: np.zeros(len(full_df), dtype=float) for model_name in MODEL_NAMES}
    fold_metric_rows: list[dict] = []

    for fold, (_, val_idx) in enumerate(folds):
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        labels = val_df["label"].to_numpy(dtype=int)
        loader_crop, loader_full = make_eval_loaders(val_df, is_test=False)

        for model_name in MODEL_NAMES:
            display_name = MODEL_DISPLAY_NAMES[model_name]
            print(f"\nOOF 4-way TTA: {display_name} fold {fold}")
            model = create_model(model_name)
            model.load_state_dict(torch.load(model_paths[model_name][fold], map_location=DEVICE))
            probs = predict_with_tta(model, loader_crop, loader_full)
            oof_probs_by_model[model_name][val_idx] = probs

            metrics = binary_metrics(labels, probs, threshold=0.5)
            fold_metric_rows.append(
                {
                    "experiment": EXPERIMENT_NAME,
                    "model": model_name,
                    "fold": fold,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                    "auroc": metrics["auroc"],
                    "pneumonia_recall": metrics["pneumonia_recall"],
                    "normal_recall": metrics["normal_recall"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                    "stopped_epoch": train_metadata[model_name][fold]["stopped_epoch"],
                    "best_epoch": train_metadata[model_name][fold]["best_epoch"],
                    "best_val_accuracy": train_metadata[model_name][fold]["best_val_accuracy"],
                    "best_val_threshold": train_metadata[model_name][fold]["best_val_threshold"],
                    "early_stopped": train_metadata[model_name][fold]["early_stopped"],
                    "checkpoint_path": train_metadata[model_name][fold]["checkpoint_path"],
                }
            )

            del model
            gc.collect()
            torch.cuda.empty_cache()

    for model_name, probs in oof_probs_by_model.items():
        out_df = pd.DataFrame(
            {
                "file_name": full_df["file_name"].values,
                "label": full_df["label"].astype(int).values,
                "fold": -1,
                "prob": probs,
            }
        )
        for fold, (_, val_idx) in enumerate(folds):
            out_df.loc[val_idx, "fold"] = fold
        out_path = os.path.join(WORK_DIR, f"oof_{model_name}{output_suffix}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    fold_metrics_name = f"fold_metrics{output_suffix}.csv" if output_suffix else "fold_metrics.csv"
    pd.DataFrame(fold_metric_rows).to_csv(os.path.join(WORK_DIR, fold_metrics_name), index=False)
    return oof_probs_by_model, fold_metric_rows


def build_nofold_validation_predictions(
    model_paths: dict[str, list[str]],
    train_metadata: dict[str, list[dict]],
    val_df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    val_df = val_df.reset_index(drop=True)
    labels = val_df["label"].to_numpy(dtype=int)
    loader_crop, loader_full = make_eval_loaders(val_df, is_test=False)
    val_probs_by_model: dict[str, np.ndarray] = {}
    metric_rows: list[dict] = []

    for model_name in MODEL_NAMES:
        display_name = MODEL_DISPLAY_NAMES[model_name]
        print(f"\nNo-fold validation 4-way TTA: {display_name}")
        model = create_model(model_name)
        model.load_state_dict(torch.load(model_paths[model_name][0], map_location=DEVICE))
        probs = predict_with_tta(model, loader_crop, loader_full)
        val_probs_by_model[model_name] = probs

        best_threshold, metrics = find_best_accuracy_threshold(labels, probs)
        metric_rows.append(
            {
                "experiment": EXPERIMENT_NAME,
                "model": model_name,
                "fold": 0,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "auroc": metrics["auroc"],
                "pneumonia_recall": metrics["pneumonia_recall"],
                "normal_recall": metrics["normal_recall"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "stopped_epoch": train_metadata[model_name][0]["stopped_epoch"],
                "best_epoch": train_metadata[model_name][0]["best_epoch"],
                "best_val_accuracy": train_metadata[model_name][0]["best_val_accuracy"],
                "best_val_threshold": train_metadata[model_name][0]["best_val_threshold"],
                "validation_best_threshold": best_threshold,
                "early_stopped": train_metadata[model_name][0]["early_stopped"],
                "checkpoint_path": train_metadata[model_name][0]["checkpoint_path"],
            }
        )

        out_df = pd.DataFrame(
            {
                "file_name": val_df["file_name"].values,
                "label": labels,
                "fold": 0,
                "prob": probs,
            }
        )
        out_path = os.path.join(WORK_DIR, f"oof_{model_name}_nofold.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(metric_rows).to_csv(os.path.join(WORK_DIR, "fold_metrics_nofold.csv"), index=False)
    return val_probs_by_model, metric_rows


def grid_search_oof_ensemble(
    oof_probs_by_model: dict[str, np.ndarray],
    file_names: np.ndarray,
    folds: np.ndarray,
    labels: np.ndarray,
    output_suffix: str = "",
) -> dict:
    labels = labels.astype(int)
    dense = oof_probs_by_model.get("densenet121", np.zeros_like(labels, dtype=float))
    conv = oof_probs_by_model.get("convnext_tiny", np.zeros_like(labels, dtype=float))
    eff = oof_probs_by_model.get("efficientnet_b3", np.zeros_like(labels, dtype=float))

    if MODEL_NAMES == ["densenet121"]:
        weight_candidates = [(1.0, 0.0, 0.0)]
    else:
        weight_candidates = []
        for w_dense in np.arange(0.0, 1.0001, 0.05):
            for w_conv in np.arange(0.0, 1.0001, 0.05):
                w_eff = 1.0 - w_dense - w_conv
                if w_eff >= -1e-9:
                    weight_candidates.append((round(float(w_dense), 2), round(float(w_conv), 2), round(float(w_eff), 2)))

    best = {
        "best_score": -1.0,
        "best_threshold": 0.5,
        "best_weights": {"densenet121": 1.0, "convnext_tiny": 0.0, "efficientnet_b3": 0.0},
        "optimization_metric": "accuracy",
        "ensemble_opt_metric": "accuracy",
    }
    best_probs = None
    for w_dense, w_conv, w_eff in weight_candidates:
        probs = w_dense * dense + w_conv * conv + w_eff * eff
        for threshold in np.arange(0.05, 0.9501, 0.001):
            preds = (probs >= threshold).astype(int)
            score = accuracy_score(labels, preds)
            if score > best["best_score"]:
                best = {
                    "best_score": float(score),
                    "best_threshold": round(float(threshold), 3),
                    "best_weights": {
                        "densenet121": float(w_dense),
                        "convnext_tiny": float(w_conv),
                        "efficientnet_b3": float(w_eff),
                    },
                    "optimization_metric": "accuracy",
                    "ensemble_opt_metric": "accuracy",
                }
                best_probs = probs.copy()

    assert best_probs is not None
    best_metrics = binary_metrics(labels, best_probs, threshold=best["best_threshold"])
    oof_preds = (best_probs >= best["best_threshold"]).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, oof_preds, labels=[0, 1]).ravel()
    config = {
        **best,
        "experiment": EXPERIMENT_NAME,
        "mode": MODE_NAME,
        "use_letterbox": USE_LETTERBOX,
        "use_clahe": USE_CLAHE,
        "use_focal_loss": USE_FOCAL_LOSS,
        "use_mixup": USE_MIXUP,
        "add_cbam_to_all": ADD_CBAM_TO_ALL,
        "best_accuracy": best_metrics["accuracy"],
        "best_f1": best_metrics["f1"],
        "best_balanced_accuracy": best_metrics["balanced_accuracy"],
        "oof_f1": best_metrics["f1"],
        "oof_auroc": best_metrics["auroc"],
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "pneumonia_recall": best_metrics["pneumonia_recall"],
        "normal_recall": best_metrics["normal_recall"],
        "predicted_positive_ratio": float(oof_preds.mean()),
        "true_positive_ratio": float(labels.mean()),
    }

    oof_ensemble_df = pd.DataFrame(
        {
            "file_name": file_names,
            "label": labels,
            "fold": folds,
            "prob": best_probs,
            "pred": oof_preds,
        }
    )
    oof_ensemble_name = f"oof_ensemble{output_suffix}.csv" if output_suffix else "oof_ensemble.csv"
    best_config_name = f"best_oof_ensemble_config{output_suffix}.json" if output_suffix else "best_oof_ensemble_config.json"
    oof_ensemble_df.to_csv(os.path.join(WORK_DIR, oof_ensemble_name), index=False)
    save_json(os.path.join(WORK_DIR, best_config_name), config)
    return config


def validate_test_images(test_df: pd.DataFrame) -> None:
    ds = KaggleChestXrayDataset(test_df, IMAGE_DIR, transform=None, is_test=True)
    missing = [name for name in test_df["file_name"].tolist() if not os.path.exists(ds.resolve_path(name))]
    print(f"test image missing count = {len(missing)}")
    if missing:
        raise FileNotFoundError(f"Missing test images. First missing files: {missing[:10]}")


def predict_test_model_average(model_paths: dict[str, list[str]], test_df: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    loader_crop, loader_full = make_eval_loaders(test_df, is_test=True)
    accumulated_probs = np.zeros(len(test_df), dtype=float)

    for model_name in MODEL_NAMES:
        weight = float(weights.get(model_name, 0.0))
        if weight <= 0:
            print(f"Skipping {model_name} final inference because weight={weight:.3f}")
            continue
        model_fold_probs = np.zeros(len(test_df), dtype=float)
        for fold, path in enumerate(model_paths[model_name]):
            print(f"Final 4-way TTA: {MODEL_DISPLAY_NAMES[model_name]} fold {fold}")
            model = create_model(model_name)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model_fold_probs += predict_with_tta(model, loader_crop, loader_full) / len(model_paths[model_name])
            del model
            gc.collect()
            torch.cuda.empty_cache()
        accumulated_probs += weight * model_fold_probs

    return accumulated_probs


def save_submission_files(test_df: pd.DataFrame, probs: np.ndarray, best_threshold: float, output_suffix: str = "") -> list[str]:
    sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    if list(test_df["file_name"]) != list(sample_df["file_name"]):
        print("test.csv file_name order differs from sample_submission.csv; aligning submission to sample order.")

    test_probs_df = pd.DataFrame({"file_name": test_df["file_name"].values, "prob": probs})
    if output_suffix:
        test_probs_name = f"test_probs{output_suffix}.csv"
        submission_prefix = f"submission{output_suffix}_th"
        final_submission_name = f"submission{output_suffix}.csv"
    else:
        test_probs_name = "test_probs.csv"
        submission_prefix = "submission_th"
        final_submission_name = "submission.csv"

    test_probs_path = os.path.join(WORK_DIR, test_probs_name)
    test_probs_df.to_csv(test_probs_path, index=False)

    prob_map = dict(zip(test_probs_df["file_name"], test_probs_df["prob"]))
    aligned_probs = sample_df["file_name"].map(prob_map)
    if aligned_probs.isna().any():
        missing = sample_df.loc[aligned_probs.isna(), "file_name"].head(10).tolist()
        raise ValueError(f"sample_submission contains file names without probabilities: {missing}")

    print(
        "test prob min/max/mean/std: "
        f"{aligned_probs.min():.6f}/{aligned_probs.max():.6f}/{aligned_probs.mean():.6f}/{aligned_probs.std():.6f}"
    )

    generated_paths = [test_probs_path]
    threshold_values = [
        best_threshold - 0.10,
        best_threshold - 0.06,
        best_threshold - 0.03,
        best_threshold - 0.02,
        best_threshold - 0.01,
        best_threshold,
        best_threshold + 0.01,
        best_threshold + 0.02,
        best_threshold + 0.03,
        best_threshold + 0.06,
        best_threshold + 0.10,
    ]
    threshold_values = sorted({round(float(np.clip(th, 0.05, 0.95)), 3) for th in threshold_values})

    final_submission_df = None
    for threshold in threshold_values:
        labels = (aligned_probs.to_numpy(dtype=float) >= threshold).astype(int)
        submission_df = pd.DataFrame({"file_name": sample_df["file_name"].values, "label": labels.astype(int)})
        validate_submission_df(sample_df, submission_df)
        normal_count = int((submission_df["label"] == 0).sum())
        pneumonia_count = int((submission_df["label"] == 1).sum())
        pneumonia_ratio = float(submission_df["label"].mean())
        print(
            f"threshold={threshold:.3f} "
            f"predicted NORMAL count={normal_count} "
            f"predicted PNEUMONIA count={pneumonia_count} "
            f"predicted PNEUMONIA ratio={pneumonia_ratio:.4f} "
            f"prob min/max/mean/std="
            f"{aligned_probs.min():.6f}/{aligned_probs.max():.6f}/{aligned_probs.mean():.6f}/{aligned_probs.std():.6f}"
        )

        path = os.path.join(WORK_DIR, f"{submission_prefix}_{threshold:.3f}.csv")
        submission_df.to_csv(path, index=False)
        generated_paths.append(path)
        if abs(threshold - best_threshold) < 1e-9:
            final_submission_df = submission_df.copy()

    if final_submission_df is None:
        labels = (aligned_probs.to_numpy(dtype=float) >= best_threshold).astype(int)
        final_submission_df = pd.DataFrame({"file_name": sample_df["file_name"].values, "label": labels.astype(int)})
        validate_submission_df(sample_df, final_submission_df)

    final_path = os.path.join(WORK_DIR, final_submission_name)
    final_submission_df.to_csv(final_path, index=False)
    validate_submission_df(sample_df, pd.read_csv(final_path))
    print(f"prediction class ratio: {final_submission_df['label'].value_counts(normalize=True).sort_index().to_dict()}")
    generated_paths.append(final_path)
    return generated_paths


def validate_submission_df(sample_df: pd.DataFrame, submission_df: pd.DataFrame) -> None:
    if list(sample_df["file_name"]) != list(submission_df["file_name"]):
        raise ValueError("submission file_name order does not match sample_submission.csv")
    if submission_df["label"].isna().any():
        raise ValueError("submission label contains NaN")
    if not np.issubdtype(submission_df["label"].dtype, np.integer):
        raise TypeError(f"submission label dtype must be int, got {submission_df['label'].dtype}")


# ==========================================
# 7. Main Execution Flow
# ==========================================
if __name__ == "__main__":
    os.makedirs(WORK_DIR, exist_ok=True)
    log_run_config()
    require_file(TRAIN_CSV, "strict train split CSV")
    require_file(VAL_CSV, "strict validation split CSV")
    require_file(TEST_CSV, "test CSV")
    require_file(SAMPLE_SUBMISSION_PATH, "sample submission CSV")
    require_file(IMAGE_DIR, "image directory")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    if not set(full_df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary with NORMAL=0 and PNEUMONIA=1.")

    group_col = select_group_column(full_df)
    log_split_info(train_df, val_df, full_df, group_col)

    model_paths: dict[str, list[str]] = {model_name: [] for model_name in MODEL_NAMES}
    train_metadata: dict[str, list[dict]] = {model_name: [] for model_name in MODEL_NAMES}
    output_suffix = "_nofold" if NO_FOLD_MODE else ""
    epoch_history_name = "epoch_history_nofold.csv" if NO_FOLD_MODE else "epoch_history.csv"
    all_epoch_history_rows: list[dict] = []

    if NO_FOLD_MODE:
        print("\nNO_FOLD_MODE=True: using train_split_strict.csv for training and val_split_strict.csv for validation.")
        print("StratifiedGroupKFold is not used in NO_FOLD_MODE.")
        for model_name in MODEL_NAMES:
            fold = 0
            f_train_df = train_df.reset_index(drop=True)
            f_val_df = val_df.reset_index(drop=True)
            train_ds = KaggleChestXrayDataset(f_train_df, IMAGE_DIR, transform=train_transform)
            val_ds = KaggleChestXrayDataset(f_val_df, IMAGE_DIR, transform=val_test_transform_crop)
            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                drop_last=True,
            )
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            path, metadata = train_fold(model_name, fold, train_loader, val_loader, output_suffix=output_suffix)
            model_paths[model_name].append(path)
            train_metadata[model_name].append(metadata)
            all_epoch_history_rows.extend(metadata["epoch_history"])

        oof_probs_by_model, fold_metric_rows = build_nofold_validation_predictions(model_paths, train_metadata, val_df)
        oof_fold_ids = np.zeros(len(val_df), dtype=int)
        oof_file_names = val_df["file_name"].to_numpy()
        oof_labels = val_df["label"].to_numpy(dtype=int)
    else:
        sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        folds = list(sgkf.split(full_df, full_df["label"], groups=full_df[group_col]))
        log_fold_distribution(full_df, folds, group_col)

        for model_name in MODEL_NAMES:
            for fold, (train_idx, val_idx) in enumerate(folds):
                f_train_df = maybe_debug_sample(full_df.iloc[train_idx].reset_index(drop=True), fold)
                f_val_df = full_df.iloc[val_idx].reset_index(drop=True)

                train_ds = KaggleChestXrayDataset(f_train_df, IMAGE_DIR, transform=train_transform)
                val_ds = KaggleChestXrayDataset(f_val_df, IMAGE_DIR, transform=val_test_transform_crop)

                train_loader = DataLoader(
                    train_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_WORKERS,
                    drop_last=True,
                )
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

                path, metadata = train_fold(model_name, fold, train_loader, val_loader, output_suffix=output_suffix)
                model_paths[model_name].append(path)
                train_metadata[model_name].append(metadata)
                all_epoch_history_rows.extend(metadata["epoch_history"])

    epoch_history_path = os.path.join(WORK_DIR, epoch_history_name)
    pd.DataFrame(all_epoch_history_rows).to_csv(epoch_history_path, index=False)
    print(f"Saved combined epoch history: {epoch_history_path}")

    if not NO_FOLD_MODE:
        oof_probs_by_model, fold_metric_rows = build_oof_predictions(
            model_paths, train_metadata, full_df, folds, output_suffix=output_suffix
        )
        oof_fold_ids = np.full(len(full_df), -1, dtype=int)
        for fold, (_, val_idx) in enumerate(folds):
            oof_fold_ids[val_idx] = fold
        oof_file_names = full_df["file_name"].to_numpy()
        oof_labels = full_df["label"].to_numpy(dtype=int)

    oof_config = grid_search_oof_ensemble(
        oof_probs_by_model,
        oof_file_names,
        oof_fold_ids,
        oof_labels,
        output_suffix=output_suffix,
    )

    if DEBUG_MODE:
        test_df = pd.read_csv(TEST_CSV)
        validate_test_images(test_df)

        print("\nDEBUG_MODE=True: smoke test completed successfully.")
        print("Stopping before final test inference and submission.csv creation.")
        print("No final submission.csv was created in DEBUG_MODE.")
        raise SystemExit(0)

    test_df = pd.read_csv(TEST_CSV)
    validate_test_images(test_df)
    test_probs = predict_test_model_average(model_paths, test_df, oof_config["best_weights"])
    generated_paths = save_submission_files(test_df, test_probs, float(oof_config["best_threshold"]), output_suffix=output_suffix)

    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    mean_accuracy_by_model = fold_metrics_df.groupby("model")["accuracy"].mean().to_dict()
    mean_f1_by_model = fold_metrics_df.groupby("model")["f1"].mean().to_dict()

    print("\nFinal log")
    print(f"used split files: {TRAIN_CSV}, {VAL_CSV}")
    print(f"used group column: {group_col}")
    print(f"leaderboard metric: accuracy")
    print(f"NO_FOLD_MODE: {NO_FOLD_MODE}")
    print(f"train rows: {len(train_df)}")
    print(f"val rows: {len(val_df)}")
    print(f"model list: {MODEL_NAMES}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"epochs: {EPOCHS}")
    print(
        "run settings: "
        f"DEBUG_MODE={DEBUG_MODE}, NO_FOLD_MODE={NO_FOLD_MODE}, EPOCHS={EPOCHS}, NUM_FOLDS={NUM_FOLDS}, IMAGE_SIZE={IMAGE_SIZE}, "
        f"MODEL_NAMES={MODEL_NAMES}, EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE}, "
        f"EARLY_STOPPING_START_EPOCH={EARLY_STOPPING_START_EPOCH}, TOP_K_CHECKPOINTS={TOP_K_CHECKPOINTS}, "
        f"USE_LETTERBOX={USE_LETTERBOX}, USE_CLAHE={USE_CLAHE}, USE_FOCAL_LOSS={USE_FOCAL_LOSS}, "
        f"USE_MIXUP={USE_MIXUP}, ADD_CBAM_TO_ALL={ADD_CBAM_TO_ALL}, ENSEMBLE_OPT_METRIC={ENSEMBLE_OPT_METRIC}"
    )
    print(f"model fold mean accuracy: {mean_accuracy_by_model}")
    print(f"model fold mean F1: {mean_f1_by_model}")
    print("fold early-stopping summary:")
    for row in fold_metrics_df[["model", "fold", "best_epoch", "stopped_epoch", "early_stopped"]].drop_duplicates().to_dict("records"):
        metadata = train_metadata[row["model"]][int(row["fold"])]
        print(
            f"- model={row['model']} fold={row['fold']} "
            f"best_epoch={row['best_epoch']} stopped_epoch={row['stopped_epoch']} "
            f"early_stopped={row['early_stopped']} "
            f"best_val_accuracy={metadata['best_val_accuracy']:.6f} "
            f"best_val_threshold={metadata['best_val_threshold']:.3f}"
        )
    print(f"best OOF weights: {oof_config['best_weights']}")
    print(f"best OOF threshold: {oof_config['best_threshold']}")
    print(f"best optimization score ({oof_config['optimization_metric']}): {oof_config['best_score']}")
    print(f"best OOF accuracy: {oof_config['best_accuracy']}")
    if NO_FOLD_MODE:
        print(f"best validation accuracy: {oof_config['best_accuracy']}")
        print(f"best threshold: {oof_config['best_threshold']}")
        print(f"best weights: {oof_config['best_weights']}")
    print(f"OOF F1 for medical analysis: {oof_config['oof_f1']}")
    print(f"test prob min/max/mean/std: {test_probs.min():.6f}/{test_probs.max():.6f}/{test_probs.mean():.6f}/{test_probs.std():.6f}")
    print("generated submission files:")
    for path in generated_paths:
        print(f"- {path}")
    print("generated artifact files:")
    artifact_paths = [epoch_history_path]
    for metadata_list in train_metadata.values():
        for metadata in metadata_list:
            artifact_paths.append(metadata["epoch_history_path"])
    artifact_paths.extend(
        [
            os.path.join(WORK_DIR, f"fold_metrics{output_suffix}.csv" if output_suffix else "fold_metrics.csv"),
            os.path.join(WORK_DIR, f"oof_ensemble{output_suffix}.csv" if output_suffix else "oof_ensemble.csv"),
            os.path.join(
                WORK_DIR,
                f"best_oof_ensemble_config{output_suffix}.json" if output_suffix else "best_oof_ensemble_config.json",
            ),
        ]
    )
    for path in artifact_paths:
        print(f"- {path}")
    print(f"\n{EXPERIMENT_NAME} completed.")
