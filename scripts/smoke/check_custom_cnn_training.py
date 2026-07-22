"""Check the Custom CNN training and checkpoint pipeline.

This script verifies that the training pipeline can run end to end. It is not
intended for model selection or performance claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ChestXrayDataset
from src.data.transforms import ResizeToTensor
from src.metrics import compute_binary_metrics
from src.models.baseline_cnn import BaselineCNN
from src.utils.seed import set_seed


LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-epoch Custom CNN training smoke test.")
    parser.add_argument("--train_csv", default="data/splits/train_duplicate_aware.csv")
    parser.add_argument("--val_csv", default="data/splits/val_duplicate_aware.csv")
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_path", default="outputs/logs/custom_cnn_smoke_train_log.csv")
    parser.add_argument("--metrics_path", default="outputs/metrics/custom_cnn_smoke_metrics.json")
    parser.add_argument("--confusion_matrix_path", default="outputs/confusion_matrices/custom_cnn_smoke_confusion_matrix.csv")
    parser.add_argument("--checkpoint_path", default="outputs/checkpoints/custom_cnn_smoke_epoch1.pt")
    parser.add_argument("--report_path", default="reports/custom_cnn_training_smoke_test.md")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(csv_path: str, image_dir: str, image_size: int, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = ChestXrayDataset(csv_path, image_dir, transform=ResizeToTensor(image_size=image_size))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator if shuffle else None,
    )


def run_train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels in loader:
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


def run_validation(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.long().to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs.cpu().tolist())
    val_loss = total_loss / max(total_items, 1)
    metrics = compute_binary_metrics(np.asarray(y_true), np.asarray(y_pred), np.asarray(y_score), positive_label=1)
    return val_loss, metrics


def save_confusion_matrix(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    df = pd.DataFrame(
        cm,
        index=["actual_NORMAL_0", "actual_PNEUMONIA_1"],
        columns=["pred_NORMAL_0", "pred_PNEUMONIA_1"],
    )
    df.to_csv(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_report(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = results["validation_metrics"]
    cm = metrics["confusion_matrix"]
    lines = [
        "# Custom CNN Training Smoke Test",
        "",
        "This one-epoch run verifies the PyTorch training pipeline only. It is not a model-selection result and must not be interpreted as clinical evidence.",
        "",
        "## Purpose",
        "",
        "- Confirm DataLoader, Custom CNN, loss, backward pass, optimizer step, validation loop, metric logging, and checkpoint saving work end to end.",
        "- Avoid hyperparameter tuning, transfer learning, Grad-CAM, or performance conclusions.",
        "",
        "## Run Configuration",
        "",
        f"- train split: `{results['config']['train_csv']}`",
        f"- validation split: `{results['config']['val_csv']}`",
        f"- model: `BaselineCNN`",
        f"- input shape: `{results['input_shape']}`",
        f"- output shape: `{results['output_shape']}`",
        f"- loss function: `CrossEntropyLoss`",
        f"- optimizer: `AdamW`",
        f"- learning rate: `{results['config']['lr']}`",
        f"- batch size: `{results['config']['batch_size']}`",
        f"- image size: `{results['config']['image_size']}`",
        f"- epochs: `{results['config']['epochs']}`",
        f"- device: `{results['device']}`",
        "",
        "## Metrics Produced",
        "",
        f"- train loss: `{results['train_loss']}`",
        f"- validation loss: `{results['val_loss']}`",
        f"- accuracy: `{metrics['accuracy']}`",
        f"- sensitivity / PNEUMONIA recall: `{metrics['recall_sensitivity']}`",
        f"- specificity / NORMAL recall: `{metrics['specificity']}`",
        f"- precision: `{metrics['precision']}`",
        f"- F1-score: `{metrics['f1']}`",
        f"- AUROC: `{metrics.get('auroc')}`",
        f"- FN count: `{metrics['fn']}`",
        f"- FP count: `{metrics['fp']}`",
        f"- confusion matrix [[TN, FP], [FN, TP]]: `{cm}`",
        "",
        "## Output Files",
        "",
        f"- checkpoint: `{results['checkpoint_path']}`",
        f"- log: `{results['log_path']}`",
        f"- metrics: `{results['metrics_path']}`",
        f"- confusion matrix: `{results['confusion_matrix_path']}`",
        "",
        "## Verification",
        "",
        f"- one epoch completed: `{results['verification']['one_epoch_completed']}`",
        f"- backward pass and optimizer step completed: `{results['verification']['optimizer_step_completed']}`",
        f"- validation loop completed: `{results['verification']['validation_loop_completed']}`",
        f"- checkpoint saved: `{results['verification']['checkpoint_saved']}`",
        f"- metrics saved: `{results['verification']['metrics_saved']}`",
        f"- no clinical validity claimed: `True`",
        "",
        "## Limitations",
        "",
        "- Patient-level identifiers are unavailable, so patient-level leakage cannot be fully excluded.",
        "- One train/test near-duplicate risk remains reported as a limitation; the test set was not modified.",
        "- One epoch is insufficient for performance conclusions.",
        "- Accuracy alone is not sufficient evidence of model quality.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs != 1:
        raise ValueError("This smoke test must run exactly 1 epoch.")

    set_seed(args.seed)
    device = choose_device(args.device)

    for output_path in [args.log_path, args.metrics_path, args.confusion_matrix_path, args.checkpoint_path, args.report_path]:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(args.train_csv, args.image_dir, args.image_size, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(args.val_csv, args.image_dir, args.image_size, args.batch_size, shuffle=False, seed=args.seed)

    first_images, _ = next(iter(train_loader))
    model = BaselineCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loss = run_train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, metrics = run_validation(model, val_loader, criterion, device)

    with torch.no_grad():
        output_shape = list(model(first_images.to(device)).shape)

    log_row = {
        "epoch": 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall_sensitivity": metrics["recall_sensitivity"],
        "specificity": metrics["specificity"],
        "f1": metrics["f1"],
        "auroc": metrics.get("auroc"),
        "fn": metrics["fn"],
        "fp": metrics["fp"],
    }
    pd.DataFrame([log_row]).to_csv(args.log_path, index=False)
    save_confusion_matrix(metrics, Path(args.confusion_matrix_path))

    config = vars(args).copy()
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 1,
        "config": config,
        "validation_metrics": json_ready(metrics),
        "smoke_test": True,
    }
    torch.save(checkpoint, args.checkpoint_path)

    results = {
        "config": config,
        "device": str(device),
        "input_shape": list(first_images.shape),
        "output_shape": output_shape,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "validation_metrics": json_ready(metrics),
        "checkpoint_path": args.checkpoint_path,
        "log_path": args.log_path,
        "metrics_path": args.metrics_path,
        "confusion_matrix_path": args.confusion_matrix_path,
        "verification": {
            "one_epoch_completed": True,
            "optimizer_step_completed": True,
            "validation_loop_completed": True,
            "checkpoint_saved": Path(args.checkpoint_path).exists(),
            "log_saved": Path(args.log_path).exists(),
            "metrics_saved": True,
            "confusion_matrix_saved": Path(args.confusion_matrix_path).exists(),
            "training_started": "smoke_test_only",
        },
    }
    Path(args.metrics_path).write_text(json.dumps(json_ready(results), indent=2), encoding="utf-8")
    write_report(Path(args.report_path), json_ready(results))
    print(json.dumps(json_ready(results), indent=2))


if __name__ == "__main__":
    main()
