"""Generate a Kaggle submission with a trained Custom CNN checkpoint.

This script loads a trained checkpoint and creates:

    /kaggle/working/submission.csv

Label mapping:
    0 = NORMAL
    1 = PNEUMONIA

This is for hackathon submission only, not clinical diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Config section
# =============================================================================

DATA_ROOT = "/kaggle/input/<DATASET_NAME>"
WORK_DIR = "/kaggle/working"

TEST_CSV = f"{DATA_ROOT}/data/test.csv"
SAMPLE_SUBMISSION_CSV = f"{DATA_ROOT}/data/sample_submission.csv"
TEST_IMAGE_DIR = f"{DATA_ROOT}/data/images"
CHECKPOINT_PATH = f"{WORK_DIR}/custom_cnn_baseline/checkpoints/best_checkpoint.pt"
OUTPUT_PATH = f"{WORK_DIR}/submission.csv"

MODEL_NAME = "custom_cnn"
IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2
SEED = 42
NORMALIZE = False
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset class
# =============================================================================


class TestXrayDataset(Dataset):
    def __init__(self, test_csv: str | Path, image_dir: str | Path, transform) -> None:
        self.df = pd.read_csv(test_csv)
        if list(self.df.columns) != ["file_name"]:
            raise ValueError(f"test.csv must contain exactly one column ['file_name'], got {self.df.columns.tolist()}")
        self.image_dir = Path(image_dir)
        self.transform = transform
        self._check_all_images_exist()

    def __len__(self) -> int:
        return len(self.df)

    def resolve_path(self, file_name: str) -> Path:
        raw = Path(str(file_name))
        candidates = [
            self.image_dir / raw,
            self.image_dir / raw.name,
            self.image_dir / "test" / raw.name,
            self.image_dir / "train" / raw.name,
            self.image_dir.parent / raw,
            self.image_dir.parent / "images" / raw,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Missing image file for file_name='{file_name}'. "
            f"Expected it under TEST_IMAGE_DIR='{self.image_dir}'. Tried first candidates: {candidates[:4]}"
        )

    def _check_all_images_exist(self) -> None:
        missing = []
        for file_name in self.df["file_name"].astype(str):
            try:
                self.resolve_path(file_name)
            except FileNotFoundError:
                missing.append(file_name)
        if missing:
            preview = missing[:10]
            raise FileNotFoundError(f"{len(missing)} test images are missing. First missing examples: {preview}")

    def __getitem__(self, index: int):
        file_name = str(self.df.iloc[index]["file_name"])
        image = Image.open(self.resolve_path(file_name)).convert("RGB")
        image = self.transform(image)
        return image, file_name


# =============================================================================
# Transform builder
# =============================================================================


class TestTransform:
    """Validation/test transform only.

    Train augmentation is intentionally not used during inference because random
    transforms would make predictions non-deterministic and inconsistent with
    validation-time evaluation.
    """

    def __init__(self, image_size: int = 224, normalize: bool = False, mean=None, std=None) -> None:
        self.image_size = image_size
        self.normalize = normalize
        self.mean = torch.tensor(mean or MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std or STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.normalize:
            tensor = (tensor - self.mean) / self.std
        return tensor


def build_transform(image_size: int, normalize: bool) -> TestTransform:
    return TestTransform(image_size=image_size, normalize=normalize, mean=MEAN, std=STD)


# =============================================================================
# Model builder
# =============================================================================


class CustomCNN(nn.Module):
    """Exact Custom CNN architecture used by kaggle/train_custom_cnn_baseline.py.

    The architecture must match training exactly. A different inference model
    structure would make checkpoint loading invalid or silently change outputs.
    """

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


def build_transfer_model(model_name: str, num_classes: int = 2, dropout: float = 0.4) -> nn.Module:
    try:
        import torchvision.models as models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"torchvision is required for model_name='{model_name}'. "
            "Use model_name='custom_cnn' or install/enable torchvision."
        ) from exc

    name = model_name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier.in_features, num_classes))
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.classifier[-1].in_features, num_classes))
    else:
        raise ValueError(f"Unsupported model_name='{model_name}'")
    return model


def build_model(model_name: str, num_classes: int = 2, dropout: float = 0.4) -> nn.Module:
    name = model_name.lower()
    if name == "custom_cnn":
        return CustomCNN(num_classes=num_classes, dropout=dropout)
    if name in {"resnet18", "resnet34", "resnet50", "densenet121", "efficientnet_b0"}:
        return build_transfer_model(name, num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported model_name='{model_name}'")


# =============================================================================
# Checkpoint loader
# =============================================================================


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Checkpoint load failure: expected checkpoint['model_state_dict'], "
        "checkpoint['state_dict'], checkpoint['model'], or a raw state_dict."
    )


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = strip_module_prefix(extract_state_dict(checkpoint))
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Architecture mismatch: checkpoint state_dict does not match the selected model architecture. "
            "Make sure model_name and model definition are exactly the same as training."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Checkpoint load failure for {checkpoint_path}: {exc}") from exc
    return checkpoint if isinstance(checkpoint, dict) else {"raw_state_dict": True}


# =============================================================================
# Inference function
# =============================================================================


def logits_to_predictions(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 2:
        raise ValueError(f"Model output must have shape [B, 2] or [B, 1], got {list(logits.shape)}")

    if logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        prob_pneumonia = probs[:, 1]
    elif logits.shape[1] == 1:
        prob_pneumonia = torch.sigmoid(logits[:, 0])
        pred = (prob_pneumonia >= 0.5).long()
    else:
        raise ValueError(f"Unsupported output shape {list(logits.shape)}. Expected [B, 2] or [B, 1].")

    return pred.cpu().numpy().astype(int), prob_pneumonia.cpu().numpy()


def run_inference(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for images, file_names in loader:
            images = images.to(device)
            logits = model(images)
            preds, prob_pneumonia = logits_to_predictions(logits)
            for file_name, pred, prob in zip(file_names, preds, prob_pneumonia):
                rows.append({"file_name": file_name, "label": int(pred), "probability_PNEUMONIA": float(prob)})
    return pd.DataFrame(rows)


# =============================================================================
# Submission writer
# =============================================================================


def write_submission(pred_df: pd.DataFrame, sample_submission_csv: str | Path, output_path: str | Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_submission_csv)
    if list(sample.columns) != ["file_name", "label"]:
        raise ValueError(f"sample_submission.csv must have columns exactly ['file_name', 'label'], got {sample.columns.tolist()}")

    # Preserve sample_submission order because Kaggle scoring expects the same
    # row order and file_name alignment as the provided template.
    pred_map = pred_df.set_index("file_name")["label"].to_dict()
    missing = [file_name for file_name in sample["file_name"].astype(str) if file_name not in pred_map]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} file_name values. First examples: {missing[:10]}")

    submission = sample[["file_name"]].copy()
    submission["label"] = submission["file_name"].map(pred_map).astype(int)
    validate_submission(submission)

    output_path = safe_output_path(output_path, sample_submission_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


def safe_output_path(output_path: str | Path, sample_submission_csv: str | Path) -> Path:
    output_path = Path(output_path)
    sample_submission_csv = Path(sample_submission_csv)

    # Kaggle mounts /kaggle/input as read-only. The generated submission must be
    # written to /kaggle/working, never back into the input dataset directory.
    if str(output_path).startswith("/kaggle/input/") or output_path == sample_submission_csv:
        fallback = Path("/kaggle/working/submission.csv")
        print(
            f"Output path '{output_path}' is read-only or points to sample_submission.csv. "
            f"Writing to '{fallback}' instead."
        )
        return fallback
    return output_path


def validate_submission(submission: pd.DataFrame) -> None:
    if list(submission.columns) != ["file_name", "label"]:
        raise ValueError(f"Invalid submission columns: {submission.columns.tolist()}. Expected ['file_name', 'label'].")
    labels = set(submission["label"].dropna().unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Invalid label values in submission: {labels}. Expected only 0 or 1.")
    if submission["label"].isna().any():
        raise ValueError("Submission contains missing label values.")


# =============================================================================
# main()
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission.csv from a trained checkpoint.")
    parser.add_argument("--test_csv", default=TEST_CSV)
    parser.add_argument("--sample_submission_csv", default=SAMPLE_SUBMISSION_CSV)
    parser.add_argument("--test_image_dir", default=TEST_IMAGE_DIR)
    parser.add_argument("--checkpoint_path", default=CHECKPOINT_PATH)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--normalize", action="store_true", default=NORMALIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    # Notebook kernels such as Colab/Kaggle inject extra arguments like
    # "-f /path/to/kernel.json". Ignore unknown args so the same script works
    # both from CLI and inside a notebook cell.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring notebook/kernel arguments: {unknown}")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model name: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output path: {args.output_path}")

    transform = build_transform(args.image_size, normalize=args.normalize)
    dataset = TestXrayDataset(args.test_csv, args.test_image_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Number of test samples: {len(dataset)}")

    model = build_model(args.model_name, num_classes=2).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint_path, device)
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        print("Checkpoint config:")
        print(json.dumps(checkpoint["config"], indent=2, default=str))

    pred_df = run_inference(model, loader, device)
    if len(pred_df) != len(dataset):
        raise ValueError(f"Prediction count mismatch: got {len(pred_df)}, expected {len(dataset)}")

    submission = write_submission(pred_df, args.sample_submission_csv, args.output_path)

    print("First few submission rows:")
    print(submission.head())
    print("Submission label distribution:")
    print(submission["label"].value_counts().sort_index())
    print(f"Saved submission: {args.output_path}")


if __name__ == "__main__":
    main()
