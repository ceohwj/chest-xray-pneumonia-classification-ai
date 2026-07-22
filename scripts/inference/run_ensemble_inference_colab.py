"""Run checkpoint ensemble inference and threshold selection in Colab.

This script performs no training or fine-tuning. It only loads the 15 phase-1
checkpoints, optimizes ensemble weights/threshold from OOF or validation
probabilities, runs 12-way TTA on test images, and writes threshold-bracketed
submission files.

Research/education portfolio project only. This is not a clinical diagnostic
system.
"""

from __future__ import annotations

import gc
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from google.colab import drive
drive.mount("/content/drive")


# ==========================================
# 1. Colab configuration
# ==========================================
PROJECT_ROOT = "/content/drive/MyDrive/xray_project"
DATA_ROOT = f"{PROJECT_ROOT}/data"
PHASE1_OUTPUT_DIR = f"{PROJECT_ROOT}/phase1_outputs"
WORK_DIR = f"{PROJECT_ROOT}/ultimate_inference_outputs"

IMAGE_DIR = f"{DATA_ROOT}/images"
VAL_CSV = f"{DATA_ROOT}/val_split_strict.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
SAMPLE_SUBMISSION_PATH = f"{DATA_ROOT}/sample_submission.csv"

MODEL_NAMES = ["densenet121", "convnext_tiny", "efficientnet_b3"]
MODEL_DISPLAY_NAMES = {
    "densenet121": "DenseNet121",
    "convnext_tiny": "ConvNeXt-Tiny",
    "efficientnet_b3": "EfficientNet-B3",
}
CHECKPOINT_PATTERNS = {
    "densenet121": ["best_densenet121_384_fold{fold}.pt"],
    "convnext_tiny": ["best_convnext_tiny_384_fold{fold}.pt", "best_convnext_384_fold{fold}.pt"],
    "efficientnet_b3": ["best_efficientnet_b3_384_fold{fold}.pt", "best_efficientnet_384_fold{fold}.pt"],
}
OOF_PATTERNS = {
    "densenet121": ["oof_densenet121.csv", "oof_densenet121_nofold.csv"],
    "convnext_tiny": ["oof_convnext_tiny.csv", "oof_convnext_tiny_nofold.csv"],
    "efficientnet_b3": ["oof_efficientnet_b3.csv", "oof_efficientnet_b3_nofold.csv"],
}

NUM_FOLDS = 5
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CLAHE = False
ADD_CBAM_TO_ALL = False
OPTIMIZE_METRIC = "accuracy"  # "accuracy" or "f1"
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.001
THRESHOLD_BRACKETS = [-0.010, -0.005, 0.0, 0.005, 0.010]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
# 2. Preprocessing and dataset
# ==========================================
class CLAHETransform:
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


class ScaleCanvas:
    """Deterministic zoom in/out on an already square image."""

    def __init__(self, scale: float, image_size: int = IMAGE_SIZE, fill: tuple[int, int, int] = (0, 0, 0)):
        self.scale = scale
        self.image_size = image_size
        self.fill = fill

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        image = img_pil.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        scaled_size = max(1, int(round(self.image_size * self.scale)))
        scaled = image.resize((scaled_size, scaled_size), Image.BILINEAR)
        if scaled_size >= self.image_size:
            left = (scaled_size - self.image_size) // 2
            top = (scaled_size - self.image_size) // 2
            return scaled.crop((left, top, left + self.image_size, top + self.image_size))
        canvas = Image.new("RGB", (self.image_size, self.image_size), self.fill)
        left = (self.image_size - scaled_size) // 2
        top = (self.image_size - scaled_size) // 2
        canvas.paste(scaled, (left, top))
        return canvas


class BrightnessFactor:
    def __init__(self, factor: float = 1.05):
        self.factor = factor

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        return ImageEnhance.Brightness(img_pil.convert("RGB")).enhance(self.factor)


class ChestXrayDataset(Dataset):
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
        return image, int(row["label"]), file_name


def build_tta_transforms() -> list[tuple[str, transforms.Compose]]:
    base_variants: list[tuple[str, list]] = [
        ("orig", []),
        ("rot_m5", [transforms.RandomRotation(degrees=(-5, -5), fill=(0, 0, 0))]),
        ("rot_p5", [transforms.RandomRotation(degrees=(5, 5), fill=(0, 0, 0))]),
        ("zoom_105", [ScaleCanvas(1.05)]),
        ("zoom_095", [ScaleCanvas(0.95)]),
        ("bright_105", [BrightnessFactor(1.05)]),
    ]
    result = []
    for name, variant_steps in base_variants:
        for flip_name, flip_steps in [("noflip", []), ("hflip", [transforms.RandomHorizontalFlip(p=1.0)])]:
            steps = []
            if USE_CLAHE:
                steps.append(CLAHETransform())
            steps.extend([PadToSquare()])
            steps.extend(variant_steps)
            if not any(isinstance(step, ScaleCanvas) for step in variant_steps):
                steps.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))
            steps.extend(flip_steps)
            steps.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
            result.append((f"{name}_{flip_name}", transforms.Compose(steps)))
    return result


TTA_TRANSFORMS = build_tta_transforms()


# ==========================================
# 3. Model definitions
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
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=None)
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
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.cbam = CBAM(in_planes=768)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cbam(self.features(x))
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


class EfficientNetB3CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b3(weights=None)
        self.features = base.features
        self.cbam = CBAM(in_planes=1536)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1536, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cbam(self.features(x))
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)


def create_model(model_name: str) -> nn.Module:
    if model_name == "densenet121":
        model = DenseNet121CBAM()
    elif model_name == "convnext_tiny":
        if ADD_CBAM_TO_ALL:
            model = ConvNeXtTinyCBAM()
        else:
            model = models.convnext_tiny(weights=None)
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, 1)
    elif model_name == "efficientnet_b3":
        if ADD_CBAM_TO_ALL:
            model = EfficientNetB3CBAM()
        else:
            model = models.efficientnet_b3(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model.to(DEVICE)


def load_state_dict_flexible(model: nn.Module, checkpoint_path: str) -> None:
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    cleaned = {}
    for key, value in state.items():
        cleaned[key.replace("module.", "", 1)] = value
    model.load_state_dict(cleaned, strict=True)


# ==========================================
# 4. IO helpers
# ==========================================
def require_path(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def resolve_checkpoint(model_name: str, fold: int) -> str:
    checked = []
    for pattern in CHECKPOINT_PATTERNS[model_name]:
        path = os.path.join(PHASE1_OUTPUT_DIR, pattern.format(fold=fold))
        checked.append(path)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Missing {MODEL_DISPLAY_NAMES[model_name]} fold {fold} checkpoint. Checked: {checked}")


def load_oof_probs() -> tuple[dict[str, pd.DataFrame], bool]:
    oof_by_model = {}
    for model_name in MODEL_NAMES:
        found_path = None
        for pattern in OOF_PATTERNS[model_name]:
            path = os.path.join(PHASE1_OUTPUT_DIR, pattern)
            if os.path.exists(path):
                found_path = path
                break
        if found_path is None:
            return {}, False
        df = pd.read_csv(found_path)
        required = {"file_name", "label", "prob"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{found_path} is missing columns: {sorted(missing)}")
        oof_by_model[model_name] = df[["file_name", "label", "prob"]].copy()
        print(f"Loaded OOF probabilities for {MODEL_DISPLAY_NAMES[model_name]}: {found_path}")
    return oof_by_model, True


# ==========================================
# 5. Inference
# ==========================================
@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, is_test: bool) -> pd.DataFrame:
    model.eval()
    probs = []
    labels = []
    file_names = []
    for batch in tqdm(loader, leave=False):
        if is_test:
            images, names = batch
        else:
            images, y, names = batch
            labels.extend(y.numpy().astype(int).tolist())
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images).view(-1)
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
        file_names.extend(list(names))
    out = pd.DataFrame({"file_name": file_names, "prob": probs})
    if not is_test:
        out["label"] = labels
    return out


def predict_12way_tta_model(model: nn.Module, df: pd.DataFrame, is_test: bool) -> pd.DataFrame:
    tta_probs = []
    base = None
    for tta_name, transform in TTA_TRANSFORMS:
        print(f"  12-way TTA pass: {tta_name}")
        dataset = ChestXrayDataset(df, IMAGE_DIR, transform=transform, is_test=is_test)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        pred_df = predict_loader(model, loader, is_test=is_test)
        if base is None:
            base_cols = ["file_name"] if is_test else ["file_name", "label"]
            base = pred_df[base_cols].copy()
        tta_probs.append(pred_df["prob"].to_numpy(dtype=float))
    assert base is not None
    base["prob"] = np.mean(np.stack(tta_probs, axis=0), axis=0)
    return base


def predict_model_family(model_name: str, df: pd.DataFrame, is_test: bool) -> pd.DataFrame:
    fold_probs = []
    base = None
    for fold in range(NUM_FOLDS):
        checkpoint_path = resolve_checkpoint(model_name, fold)
        print(f"\nLoading {MODEL_DISPLAY_NAMES[model_name]} fold {fold}: {checkpoint_path}")
        model = create_model(model_name)
        load_state_dict_flexible(model, checkpoint_path)
        pred_df = predict_12way_tta_model(model, df, is_test=is_test)
        if base is None:
            base_cols = ["file_name"] if is_test else ["file_name", "label"]
            base = pred_df[base_cols].copy()
        fold_probs.append(pred_df["prob"].to_numpy(dtype=float))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert base is not None
    base["prob"] = np.mean(np.stack(fold_probs, axis=0), axis=0)
    return base


def build_validation_probs(val_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    print("\nOOF CSV files were not found for all models.")
    print("Falling back to 5-fold checkpoint average predictions on val_split_strict.csv for weight/threshold optimization.")
    probs_by_model = {}
    for model_name in MODEL_NAMES:
        print(f"\nValidation inference for {MODEL_DISPLAY_NAMES[model_name]}")
        pred_df = predict_model_family(model_name, val_df, is_test=False)
        out_path = os.path.join(WORK_DIR, f"validation_probs_{model_name}_12tta.csv")
        pred_df.to_csv(out_path, index=False)
        print(f"Saved validation probabilities: {out_path}")
        probs_by_model[model_name] = pred_df
    return probs_by_model


# ==========================================
# 6. Nelder-Mead ensemble optimization
# ==========================================
def softmax_weights(raw_weights: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_weights, dtype=float)
    raw = raw - np.max(raw)
    exp = np.exp(raw)
    return exp / exp.sum()


def merge_model_probs(probs_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = probs_by_model[MODEL_NAMES[0]][["file_name", "label"]].copy()
    for model_name in MODEL_NAMES:
        temp = probs_by_model[model_name][["file_name", "prob"]].rename(columns={"prob": f"prob_{model_name}"})
        base = base.merge(temp, on="file_name", how="inner")
    if len(base) != len(probs_by_model[MODEL_NAMES[0]]):
        raise ValueError("OOF/validation probability files do not align by file_name.")
    return base


def best_threshold_for_probs(labels: np.ndarray, probs: np.ndarray, metric: str) -> tuple[float, float]:
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-12, THRESHOLD_STEP)
    best_score = -1.0
    best_threshold = 0.5
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        if metric == "accuracy":
            score = accuracy_score(labels, preds)
        elif metric == "f1":
            score = f1_score(labels, preds, zero_division=0)
        else:
            raise ValueError("OPTIMIZE_METRIC must be 'accuracy' or 'f1'.")
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def optimize_weights_nelder_mead(merged: pd.DataFrame) -> dict:
    labels = merged["label"].to_numpy(dtype=int)
    prob_matrix = np.stack([merged[f"prob_{model_name}"].to_numpy(dtype=float) for model_name in MODEL_NAMES], axis=1)

    def objective(raw_weights: np.ndarray) -> float:
        weights = softmax_weights(raw_weights)
        probs = prob_matrix @ weights
        _, score = best_threshold_for_probs(labels, probs, OPTIMIZE_METRIC)
        return -score

    result = minimize(
        objective,
        x0=np.zeros(len(MODEL_NAMES), dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 250, "xatol": 1e-5, "fatol": 1e-5, "disp": True},
    )
    weights_arr = softmax_weights(result.x)
    ensemble_probs = prob_matrix @ weights_arr
    best_threshold, best_score = best_threshold_for_probs(labels, ensemble_probs, OPTIMIZE_METRIC)
    preds = (ensemble_probs >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    try:
        auroc = roc_auc_score(labels, ensemble_probs)
    except ValueError:
        auroc = float("nan")

    weights = {model_name: float(weight) for model_name, weight in zip(MODEL_NAMES, weights_arr)}
    config = {
        "optimizer": "scipy.optimize.minimize(method='Nelder-Mead')",
        "optimize_metric": OPTIMIZE_METRIC,
        "best_score": float(best_score),
        "best_accuracy": float(accuracy_score(labels, preds)),
        "best_f1": float(f1_score(labels, preds, zero_division=0)),
        "best_threshold": float(best_threshold),
        "best_weights": weights,
        "auroc": float(auroc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "pneumonia_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "normal_recall": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "predicted_positive_ratio": float(preds.mean()),
        "true_positive_ratio": float(labels.mean()),
        "nelder_mead_success": bool(result.success),
        "nelder_mead_message": str(result.message),
    }

    ensemble_df = merged[["file_name", "label"]].copy()
    ensemble_df["prob"] = ensemble_probs
    ensemble_df["pred"] = preds
    return {"config": config, "ensemble_df": ensemble_df}


# ==========================================
# 7. Test ensemble and submissions
# ==========================================
def predict_test_probs(test_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    final_probs = np.zeros(len(test_df), dtype=float)
    for model_name in MODEL_NAMES:
        print(f"\nTest inference for {MODEL_DISPLAY_NAMES[model_name]}")
        pred_df = predict_model_family(model_name, test_df, is_test=True)
        model_prob_path = os.path.join(WORK_DIR, f"test_probs_{model_name}_12tta.csv")
        pred_df.to_csv(model_prob_path, index=False)
        print(f"Saved model test probabilities: {model_prob_path}")
        final_probs += float(weights[model_name]) * pred_df["prob"].to_numpy(dtype=float)
    out = pd.DataFrame({"file_name": test_df["file_name"].values, "prob": final_probs})
    out_path = os.path.join(WORK_DIR, "test_probs_ultimate_12tta_nelder_mead.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved final test probabilities: {out_path}")
    return out


def bracket_thresholds(best_threshold: float) -> list[float]:
    thresholds = []
    for delta in THRESHOLD_BRACKETS:
        threshold = float(np.clip(best_threshold + delta, THRESHOLD_MIN, THRESHOLD_MAX))
        threshold = round(threshold, 3)
        if threshold not in thresholds:
            thresholds.append(threshold)
    return thresholds


def save_submissions(test_probs: pd.DataFrame, best_threshold: float) -> list[str]:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    merged = sample[["file_name"]].merge(test_probs, on="file_name", how="left")
    if merged["prob"].isna().any():
        missing = merged.loc[merged["prob"].isna(), "file_name"].head(10).tolist()
        raise ValueError(f"Missing test probabilities for sample submission files: {missing}")

    generated = []
    thresholds = bracket_thresholds(best_threshold)
    for threshold in thresholds:
        preds = (merged["prob"].to_numpy(dtype=float) >= threshold).astype(int)
        submission = sample.copy()
        submission["label"] = preds
        path = os.path.join(WORK_DIR, f"submission_th_{threshold:.3f}.csv")
        submission.to_csv(path, index=False)
        generated.append(path)
        print(
            f"Saved {path} | threshold={threshold:.3f} | "
            f"NORMAL={(preds == 0).sum()} | PNEUMONIA={(preds == 1).sum()} | "
            f"PNEUMONIA ratio={preds.mean():.4f} | "
            f"prob min/max/mean/std={merged['prob'].min():.6f}/"
            f"{merged['prob'].max():.6f}/{merged['prob'].mean():.6f}/{merged['prob'].std():.6f}"
        )
    canonical_path = os.path.join(WORK_DIR, "submission.csv")
    best_preds = (merged["prob"].to_numpy(dtype=float) >= best_threshold).astype(int)
    canonical = sample.copy()
    canonical["label"] = best_preds
    canonical.to_csv(canonical_path, index=False)
    generated.append(canonical_path)
    print(f"Saved canonical best-threshold submission: {canonical_path}")
    return generated


# ==========================================
# 8. Main
# ==========================================
def main() -> None:
    os.makedirs(WORK_DIR, exist_ok=True)
    for path, label in [
        (IMAGE_DIR, "image directory"),
        (VAL_CSV, "validation CSV"),
        (TEST_CSV, "test CSV"),
        (SAMPLE_SUBMISSION_PATH, "sample submission"),
        (PHASE1_OUTPUT_DIR, "phase-1 output directory"),
    ]:
        require_path(path, label)
    for model_name in MODEL_NAMES:
        for fold in range(NUM_FOLDS):
            resolve_checkpoint(model_name, fold)

    print("\nUltimate inference configuration")
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"PHASE1_OUTPUT_DIR={PHASE1_OUTPUT_DIR}")
    print(f"WORK_DIR={WORK_DIR}")
    print(f"DEVICE={DEVICE}")
    print(f"MODEL_NAMES={MODEL_NAMES}")
    print(f"NUM_FOLDS={NUM_FOLDS}, IMAGE_SIZE={IMAGE_SIZE}, TTA ways={len(TTA_TRANSFORMS)}")
    print(f"OPTIMIZE_METRIC={OPTIMIZE_METRIC}")

    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    oof_by_model, has_oof = load_oof_probs()
    if has_oof:
        probs_by_model = oof_by_model
        print("\nUsing saved OOF probabilities for Nelder-Mead weight/threshold optimization.")
    else:
        probs_by_model = build_validation_probs(val_df)

    merged = merge_model_probs(probs_by_model)
    opt = optimize_weights_nelder_mead(merged)
    config = opt["config"]
    oof_ensemble = opt["ensemble_df"]

    oof_path = os.path.join(WORK_DIR, "oof_ensemble_ultimate_nelder_mead.csv")
    config_path = os.path.join(WORK_DIR, "ultimate_ensemble_config.json")
    oof_ensemble.to_csv(oof_path, index=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\nSaved optimized OOF/validation ensemble: {oof_path}")
    print(f"Saved optimized config: {config_path}")
    print(f"Best weights: {config['best_weights']}")
    print(f"Best threshold: {config['best_threshold']:.6f}")
    print(f"Best {OPTIMIZE_METRIC}: {config['best_score']:.6f}")
    print(f"Accuracy: {config['best_accuracy']:.6f}, F1: {config['best_f1']:.6f}, AUROC: {config['auroc']:.6f}")
    print(f"Confusion matrix: {config['confusion_matrix']}")

    test_probs = predict_test_probs(test_df, config["best_weights"])
    generated = save_submissions(test_probs, float(config["best_threshold"]))

    print("\nUltimate inference completed.")
    print(f"Generated submission files: {generated}")


if __name__ == "__main__":
    main()
