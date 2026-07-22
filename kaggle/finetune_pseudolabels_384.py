"""Fine-tune the 384px ensemble with strict pseudo labels on Kaggle.

Research/education portfolio project only. This is not a clinical diagnostic
system.

Core safeguards:
- Use only high-confidence pseudo labels: prob < 0.05 or prob > 0.95.
- Fine-tune very gently: head LR 1e-5, backbone LR 1e-6, 3 epochs.
- Match phase-1 preprocessing: letterbox PadToSquare, CLAHE, MixUp,
  BinaryFocalLoss(alpha=0.5, gamma=1.5).
- Never silently fallback to black images.
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
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ==========================================
# 1. Kaggle configuration
# ==========================================
EXPERIMENT_NAME = "phase2_pseudo_finetune_384_high_conf_focal_letterbox"

DATA_ROOT = "/kaggle/input/datasets/hyunwoo11/chest-xray-ai"
IMAGE_DIR = f"{DATA_ROOT}/data/images"
TRAIN_CSV = f"{DATA_ROOT}/train_split_strict.csv"
VAL_CSV = f"{DATA_ROOT}/val_split_strict.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/datasets/hyunwoo11/submission/sample_submission.csv"
WORK_DIR = "/kaggle/working"

# Set this to the Kaggle dataset path that contains phase-1 .pt checkpoints
# and test_probs.csv. Example after Add Data:
# PHASE1_OUTPUT_DIR = "/kaggle/input/my-phase1-ensemble-results"
PHASE1_OUTPUT_DIR = "/kaggle/input/나의-1차-학습-결과물-데이터셋-이름"
TEST_PROBS_CSV = f"{PHASE1_OUTPUT_DIR}/test_probs.csv"

MODEL_NAMES = ["densenet121", "convnext_tiny", "efficientnet_b3"]
NUM_FOLDS = 5
IMAGE_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS_FT = 3
LR_HEAD = 1e-5
LR_BACKBONE = 1e-6
WEIGHT_DECAY = 1e-4
PSEUDO_LOSS_WEIGHT = 0.35

PSEUDO_LOW_THRESHOLD = 0.05
PSEUDO_HIGH_THRESHOLD = 0.95

USE_CLAHE = True
USE_MIXUP = True
MIXUP_ALPHA = 0.1
FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 1.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CHECKPOINT_PATTERNS = {
    "densenet121": ["best_densenet121_384_fold{fold}.pt"],
    "convnext_tiny": ["best_convnext_tiny_384_fold{fold}.pt", "best_convnext_384_fold{fold}.pt"],
    "efficientnet_b3": ["best_efficientnet_b3_384_fold{fold}.pt", "best_efficientnet_384_fold{fold}.pt"],
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
# 2. Transforms and dataset
# ==========================================
class PadToSquare:
    def __init__(self, fill: tuple[int, int, int] = (0, 0, 0)):
        self.fill = fill

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        image = img_pil.convert("RGB")
        width, height = image.size
        side = max(width, height)
        padded = Image.new("RGB", (side, side), self.fill)
        padded.paste(image, ((side - width) // 2, (side - height) // 2))
        return padded


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        img_np = np.array(img_pil.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        cl = self.clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


def build_transform(train: bool) -> transforms.Compose:
    steps = []
    if USE_CLAHE:
        # Apply CLAHE before letterbox/resize so black padding is not enhanced.
        steps.append(CLAHETransform())
    steps.extend([PadToSquare(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))])
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.08, contrast=0.08),
            ]
        )
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None, is_test: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        if "file_name" not in self.df.columns:
            raise ValueError("DataFrame must contain file_name column.")
        if not is_test and "label" not in self.df.columns:
            raise ValueError("Training/validation DataFrame must contain label column.")

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, file_name: str) -> str:
        paths = [
            os.path.join(self.image_dir, file_name),
            os.path.join(self.image_dir, "train", file_name),
            os.path.join(self.image_dir, "test", file_name),
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Missing image file: {file_name}. Checked under {self.image_dir}")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_name = row["file_name"]
        image = Image.open(self.resolve_path(file_name)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.is_test:
            return image, file_name
        label = torch.tensor(float(row["label"]), dtype=torch.float32).view(1)
        weight = torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32).view(1)
        return image, label, weight, file_name


# ==========================================
# 3. Models and loss
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DenseNet121CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=None)
        self.features = base.features
        self.cbam = CBAM(1024)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.features(x), inplace=True)
        features = self.cbam(features)
        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        return self.classifier(torch.flatten(pooled, 1))


def create_model(model_name: str) -> nn.Module:
    if model_name == "densenet121":
        return DenseNet121CBAM()
    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 1)
        return model
    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        return model
    raise ValueError(f"Unknown model: {model_name}")


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 1.5, reduction: str = "none"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_factor * (1.0 - pt).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    mixed_labels = lam * labels + (1.0 - lam) * labels[index]
    mixed_weights = lam * weights + (1.0 - lam) * weights[index]
    return mixed_images, mixed_labels, mixed_weights


def get_param_groups(model_name: str, model: nn.Module):
    if model_name == "densenet121":
        backbone_params = model.features.parameters()
        head_params = list(model.classifier.parameters()) + list(model.cbam.parameters())
    else:
        backbone_params = model.features.parameters()
        head_params = model.classifier.parameters()
    return [
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ]


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format: {path}")


def load_model(model_name: str, checkpoint_path: str | Path) -> nn.Module:
    model = create_model(model_name).to(DEVICE)
    state_dict = load_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch for {checkpoint_path}: missing={missing[:8]}, unexpected={unexpected[:8]}")
    return model


# ==========================================
# 4. Files, pseudo labels, metrics
# ==========================================
def require_file(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required {description} not found: {path}")


def find_checkpoint(model_name: str, fold: int) -> str:
    for pattern in CHECKPOINT_PATTERNS[model_name]:
        phase1_path = os.path.join(PHASE1_OUTPUT_DIR, pattern.format(fold=fold))
        if os.path.exists(phase1_path):
            return phase1_path
    expected = [os.path.join(PHASE1_OUTPUT_DIR, pattern.format(fold=fold)) for pattern in CHECKPOINT_PATTERNS[model_name]]
    raise FileNotFoundError(f"1차 학습 가중치를 찾을 수 없습니다: {expected}")


def probability_column(df: pd.DataFrame) -> str:
    for col in ["prob", "probability", "probability_PNEUMONIA", "pneumonia_prob", "label_prob"]:
        if col in df.columns:
            return col
    raise ValueError("Pseudo-label source must contain a probability column such as prob or probability_PNEUMONIA.")


def build_pseudo_labels(test_df: pd.DataFrame) -> pd.DataFrame:
    require_file(TEST_PROBS_CSV, "phase-1 test probability CSV")
    prob_df = pd.read_csv(TEST_PROBS_CSV)
    prob_col = probability_column(prob_df)
    if "file_name" not in prob_df.columns:
        raise ValueError(f"{TEST_PROBS_CSV} must contain file_name column.")

    merged = test_df[["file_name"]].merge(prob_df[["file_name", prob_col]], on="file_name", how="left")
    if merged[prob_col].isna().any():
        missing = merged.loc[merged[prob_col].isna(), "file_name"].head(10).tolist()
        raise ValueError(f"Missing phase-1 probabilities for test files: {missing}")

    merged = merged.rename(columns={prob_col: "prob"})
    confident = merged[(merged["prob"] < PSEUDO_LOW_THRESHOLD) | (merged["prob"] > PSEUDO_HIGH_THRESHOLD)].copy()
    confident["label"] = (confident["prob"] > PSEUDO_HIGH_THRESHOLD).astype(int)
    confident["pseudo_confidence"] = np.maximum(confident["prob"], 1.0 - confident["prob"])
    confident["sample_weight"] = PSEUDO_LOSS_WEIGHT

    out_all = os.path.join(WORK_DIR, "phase2_phase1_test_probs_used.csv")
    out_pseudo = os.path.join(WORK_DIR, "phase2_pseudo_labels_high_conf.csv")
    merged.to_csv(out_all, index=False)
    confident[["file_name", "label", "prob", "pseudo_confidence", "sample_weight"]].to_csv(out_pseudo, index=False)
    print(f"Loaded phase-1 probabilities: {TEST_PROBS_CSV}")
    print(f"High-confidence pseudo labels: {len(confident)} / {len(merged)}")
    print(confident["label"].value_counts().sort_index().to_string())
    print(f"Saved pseudo labels: {out_pseudo}")
    if confident.empty:
        raise ValueError("No high-confidence pseudo labels selected. Do not fine-tune on uncertain pseudo labels.")
    return confident[["file_name", "label", "sample_weight"]].copy()


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": auroc,
        "normal_recall": float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
        "pneumonia_recall": float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_accuracy_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict]:
    best_t = 0.5
    best_acc = -1.0
    for threshold in np.arange(0.05, 0.9501, 0.001):
        acc = accuracy_score(labels, (probs >= threshold).astype(int))
        if acc > best_acc:
            best_acc = float(acc)
            best_t = round(float(threshold), 3)
    return best_t, binary_metrics(labels, probs, best_t)


# ==========================================
# 5. Training and inference
# ==========================================
def make_train_loader(train_df: pd.DataFrame, pseudo_df: pd.DataFrame) -> DataLoader:
    real_df = train_df.copy()
    real_df["sample_weight"] = 1.0
    expanded = pd.concat([real_df, pseudo_df], ignore_index=True)
    expanded.to_csv(os.path.join(WORK_DIR, "phase2_expanded_train_df.csv"), index=False)
    print(f"Expanded train rows: {len(expanded)} = real {len(real_df)} + pseudo {len(pseudo_df)}")
    return DataLoader(
        ChestXrayDataset(expanded, IMAGE_DIR, build_transform(train=True), is_test=False),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def make_eval_loader(df: pd.DataFrame, is_test: bool) -> DataLoader:
    return DataLoader(
        ChestXrayDataset(df, IMAGE_DIR, build_transform(train=False), is_test=is_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    running = 0.0
    weight_sum = 0.0
    for images, labels, weights, _file_names in tqdm(loader, desc="phase2 train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        weights = weights.to(DEVICE)
        if USE_MIXUP:
            images, labels, weights = mixup_batch(images, labels, weights)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss_raw = criterion(logits, labels)
        loss = (loss_raw * weights).sum() / weights.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * float(weights.sum().item())
        weight_sum += float(weights.sum().item())
    return running / max(weight_sum, 1.0)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, is_test: bool) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        if is_test:
            images, file_names = batch
            labels = None
        else:
            images, labels, _weights, file_names = batch
        images = images.to(DEVICE)
        probs = torch.sigmoid(model(images)).view(-1).detach().cpu().numpy()
        probs_flip = torch.sigmoid(model(torch.flip(images, dims=[3]))).view(-1).detach().cpu().numpy()
        probs = (probs + probs_flip) / 2.0
        for i, file_name in enumerate(file_names):
            row = {"file_name": file_name, "prob": float(probs[i])}
            if labels is not None:
                row["label"] = int(labels[i].item())
            rows.append(row)
    return pd.DataFrame(rows)


def fine_tune_all(train_df: pd.DataFrame, val_df: pd.DataFrame, pseudo_df: pd.DataFrame) -> pd.DataFrame:
    criterion = BinaryFocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="none")
    train_loader = make_train_loader(train_df, pseudo_df)
    val_loader = make_eval_loader(val_df, is_test=False)
    rows = []
    checkpoint_dir = os.path.join(WORK_DIR, "phase2_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for model_name in MODEL_NAMES:
        for fold in range(NUM_FOLDS):
            source_path = find_checkpoint(model_name, fold)
            print(f"\nFine-tuning {model_name} fold {fold} from {source_path}")
            model = load_model(model_name, source_path)
            optimizer = optim.AdamW(get_param_groups(model_name, model), weight_decay=WEIGHT_DECAY)

            best_acc = -1.0
            best_path = os.path.join(checkpoint_dir, f"phase2_{model_name}_384_fold{fold}.pt")
            for epoch in range(1, EPOCHS_FT + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
                val_pred = predict_loader(model, val_loader, is_test=False)
                threshold, metrics = best_accuracy_threshold(
                    val_pred["label"].to_numpy(dtype=int),
                    val_pred["prob"].to_numpy(dtype=float),
                )
                row = {
                    "model": model_name,
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "best_threshold": threshold,
                    **metrics,
                    "source_checkpoint": source_path,
                    "checkpoint_path": best_path,
                    "lr_head": LR_HEAD,
                    "lr_backbone": LR_BACKBONE,
                }
                rows.append(row)
                print(
                    f"{model_name} fold={fold} epoch={epoch}/{EPOCHS_FT} "
                    f"loss={train_loss:.5f} acc={metrics['accuracy']:.4f} "
                    f"f1={metrics['f1']:.4f} auroc={metrics['auroc']:.4f} "
                    f"thr={threshold:.3f} fn={metrics['fn']} fp={metrics['fp']}"
                )
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    torch.save(model.state_dict(), best_path)

            del model
            gc.collect()
            torch.cuda.empty_cache()

    log_df = pd.DataFrame(rows)
    log_df.to_csv(os.path.join(WORK_DIR, "phase2_pseudo_finetune_log.csv"), index=False)
    return log_df


def final_inference(test_df: pd.DataFrame) -> pd.DataFrame:
    probs_by_model = {}
    for model_name in MODEL_NAMES:
        fold_probs = []
        for fold in range(NUM_FOLDS):
            path = os.path.join(WORK_DIR, "phase2_checkpoints", f"phase2_{model_name}_384_fold{fold}.pt")
            model = load_model(model_name, path)
            pred = predict_loader(model, make_eval_loader(test_df, is_test=True), is_test=True)
            fold_probs.append(pred["prob"].to_numpy(dtype=float))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        probs_by_model[model_name] = np.stack(fold_probs).mean(axis=0)

    probs = np.mean([probs_by_model[name] for name in MODEL_NAMES], axis=0)
    out = test_df[["file_name"]].copy()
    out["prob"] = probs
    out.to_csv(os.path.join(WORK_DIR, "phase2_test_probs.csv"), index=False)
    return out


def validation_ensemble_threshold(val_df: pd.DataFrame) -> tuple[float, dict]:
    probs_by_model = {}
    val_loader = make_eval_loader(val_df, is_test=False)
    labels = val_df["label"].to_numpy(dtype=int)
    for model_name in MODEL_NAMES:
        fold_probs = []
        for fold in range(NUM_FOLDS):
            path = os.path.join(WORK_DIR, "phase2_checkpoints", f"phase2_{model_name}_384_fold{fold}.pt")
            model = load_model(model_name, path)
            pred = predict_loader(model, val_loader, is_test=False)
            fold_probs.append(pred["prob"].to_numpy(dtype=float))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        probs_by_model[model_name] = np.stack(fold_probs).mean(axis=0)

    ensemble_probs = np.mean([probs_by_model[name] for name in MODEL_NAMES], axis=0)
    val_probs_df = val_df[["file_name", "label"]].copy()
    val_probs_df["prob"] = ensemble_probs
    val_probs_df.to_csv(os.path.join(WORK_DIR, "phase2_validation_ensemble_probs.csv"), index=False)
    threshold, metrics = best_accuracy_threshold(labels, ensemble_probs)
    with open(os.path.join(WORK_DIR, "phase2_validation_ensemble_config.json"), "w", encoding="utf-8") as f:
        json.dump({"best_threshold": threshold, **metrics}, f, indent=2)
    print(f"Phase2 validation ensemble best accuracy={metrics['accuracy']:.4f} at threshold={threshold:.3f}")
    return threshold, metrics


def create_submissions(test_probs: pd.DataFrame, threshold: float) -> list[str]:
    sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    merged = sample[["file_name"]].merge(test_probs, on="file_name", how="left")
    if merged["prob"].isna().any():
        missing = merged.loc[merged["prob"].isna(), "file_name"].head(10).tolist()
        raise ValueError(f"Missing final probabilities for sample rows: {missing}")

    generated = []
    threshold_values = sorted({round(float(np.clip(t, 0.05, 0.95)), 3) for t in [threshold - 0.05, threshold, threshold + 0.05, 0.5]})
    for t in threshold_values:
        submission = sample[["file_name"]].copy()
        submission["label"] = (merged["prob"].to_numpy(dtype=float) >= t).astype(int)
        path = os.path.join(WORK_DIR, f"phase2_submission_th_{t:.3f}.csv")
        submission.to_csv(path, index=False)
        generated.append(path)
        print(f"threshold={t:.3f} pneumonia_ratio={submission['label'].mean():.4f}")

    final = sample[["file_name"]].copy()
    final["label"] = (merged["prob"].to_numpy(dtype=float) >= threshold).astype(int)
    final_path = os.path.join(WORK_DIR, "submission.csv")
    final.to_csv(final_path, index=False)
    generated.append(final_path)
    return generated


def main() -> None:
    os.makedirs(WORK_DIR, exist_ok=True)
    for path, desc in [
        (TRAIN_CSV, "train split"),
        (VAL_CSV, "validation split"),
        (TEST_CSV, "test CSV"),
        (SAMPLE_SUBMISSION_PATH, "sample submission"),
        (IMAGE_DIR, "image directory"),
        (PHASE1_OUTPUT_DIR, "phase-1 output directory"),
        (TEST_PROBS_CSV, "phase-1 test probability CSV"),
    ]:
        require_file(path, desc)

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Device: {DEVICE}")
    print(f"PHASE1_OUTPUT_DIR={PHASE1_OUTPUT_DIR}")
    print(f"TEST_PROBS_CSV={TEST_PROBS_CSV}")
    print(f"EPOCHS_FT={EPOCHS_FT}, LR_HEAD={LR_HEAD}, LR_BACKBONE={LR_BACKBONE}")
    print(f"Pseudo filter: prob < {PSEUDO_LOW_THRESHOLD} or prob > {PSEUDO_HIGH_THRESHOLD}")
    print(f"Preprocess: PadToSquare letterbox, USE_CLAHE={USE_CLAHE}, USE_MIXUP={USE_MIXUP}")
    print(f"Loss: BinaryFocalLoss(alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    pseudo_df = build_pseudo_labels(test_df)
    log_df = fine_tune_all(train_df, val_df, pseudo_df)

    best_threshold, val_ensemble_metrics = validation_ensemble_threshold(val_df)
    test_probs = final_inference(test_df)
    generated = create_submissions(test_probs, best_threshold)

    config = {
        "experiment": EXPERIMENT_NAME,
        "epochs_ft": EPOCHS_FT,
        "lr_head": LR_HEAD,
        "lr_backbone": LR_BACKBONE,
        "pseudo_low_threshold": PSEUDO_LOW_THRESHOLD,
        "pseudo_high_threshold": PSEUDO_HIGH_THRESHOLD,
        "pseudo_rows": int(len(pseudo_df)),
        "best_validation_threshold": best_threshold,
        "validation_ensemble_metrics": val_ensemble_metrics,
        "generated_files": generated,
    }
    with open(os.path.join(WORK_DIR, "phase2_pseudo_finetune_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nPhase 2 pseudo fine-tuning complete.")
    print(f"Best validation threshold: {best_threshold}")
    print("Generated files:")
    for path in generated:
        print(f"- {path}")


if __name__ == "__main__":
    main()
