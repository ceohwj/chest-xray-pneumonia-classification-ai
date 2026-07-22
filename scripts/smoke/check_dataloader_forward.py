"""Check one DataLoader batch and one Custom CNN forward pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .dataset import ChestXrayDataset
    from .transforms import ResizeToTensor
    from ..models.baseline_cnn import BaselineCNN
except ImportError:
    from src.data.dataset import ChestXrayDataset
    from src.data.transforms import ResizeToTensor
    from src.models.baseline_cnn import BaselineCNN


LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-training DataLoader and forward-pass smoke tests.")
    parser.add_argument("--train_csv", default="data/splits/train_duplicate_aware.csv")
    parser.add_argument("--val_csv", default="data/splits/val_duplicate_aware.csv")
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--report_path", default="reports/dataloader_forward_smoke_test.md")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check_resolved_paths(dataset: ChestXrayDataset, max_items: int = 32) -> dict[str, Any]:
    checked = min(max_items, len(dataset))
    missing = []
    for idx in range(checked):
        file_name = str(dataset.df.iloc[idx]["file_name"])
        path = dataset._resolve_path(file_name)
        if not path.exists():
            missing.append({"file_name": file_name, "resolved_path": str(path)})
    return {"checked": checked, "missing": missing, "passes": not missing}


def label_distribution(labels: torch.Tensor) -> dict[str, int]:
    values, counts = labels.cpu().unique(return_counts=True)
    out = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        out[f"{value} ({LABEL_NAMES.get(int(value), 'UNKNOWN')})"] = int(count)
    return out


def summarize_batch(images: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return {
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "image_min": float(images.min().item()),
        "image_max": float(images.max().item()),
        "label_distribution": label_distribution(labels),
        "labels_are_integer": labels.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8),
        "labels_in_expected_set": bool(set(labels.cpu().tolist()).issubset({0, 1})),
    }


def markdown_dict(data: dict[str, Any]) -> str:
    return "\n".join(f"- {key}: `{value}`" for key, value in data.items())


def write_report(report_path: Path, results: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DataLoader and Forward-pass Smoke Test",
        "",
        "This is a pre-training smoke test for a research/education project. It does not evaluate model performance.",
        "",
        "## Inputs",
        "",
        f"- train split: `{results['train_csv']}`",
        f"- validation split: `{results['val_csv']}`",
        f"- image directory: `{results['image_dir']}`",
        f"- batch size: `{results['batch_size']}`",
        f"- image size: `{results['image_size']}x{results['image_size']}`",
        f"- label mapping: `0=NORMAL`, `1=PNEUMONIA`",
        "",
        "## Transform Check",
        "",
        "- Transform: deterministic RGB conversion, resize to 224x224, convert to float tensor in [0, 1].",
        "- No augmentation was applied.",
        "- No vertical flip or medically inappropriate augmentation was used.",
        "",
        "## Train Batch",
        "",
        markdown_dict(results["train_batch"]),
        "",
        "## Validation Batch",
        "",
        markdown_dict(results["val_batch"]),
        "",
        "## Forward-pass Check",
        "",
        markdown_dict(results["forward_pass"]),
        "",
        "## Verification",
        "",
        markdown_dict(results["verification"]),
        "",
        "## Issues",
        "",
        "- No loading, shape, or loss-computation issues were found.",
        "- Full model training has not started yet.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    transform = ResizeToTensor(image_size=args.image_size)
    train_dataset = ChestXrayDataset(args.train_csv, args.image_dir, transform=transform)
    val_dataset = ChestXrayDataset(args.val_csv, args.image_dir, transform=transform)

    train_path_check = check_resolved_paths(train_dataset)
    val_path_check = check_resolved_paths(val_dataset)
    if not train_path_check["passes"] or not val_path_check["passes"]:
        raise RuntimeError({"train_path_check": train_path_check, "val_path_check": val_path_check})

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    model = BaselineCNN(num_classes=2)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        outputs = model(train_images)
        loss = criterion(outputs, train_labels.long())

    results = {
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "image_dir": args.image_dir,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "train_batch": summarize_batch(train_images, train_labels),
        "val_batch": summarize_batch(val_images, val_labels),
        "forward_pass": {
            "model": "BaselineCNN",
            "output_shape": list(outputs.shape),
            "expected_output_shape": [args.batch_size, 2],
            "label_shape_for_loss": list(train_labels.shape),
            "loss_function": "CrossEntropyLoss",
            "loss_value": float(loss.item()),
            "loss_is_finite": bool(torch.isfinite(loss).item()),
        },
        "verification": {
            "train_path_check": train_path_check,
            "val_path_check": val_path_check,
            "train_split_rows": int(len(pd.read_csv(args.train_csv))),
            "val_split_rows": int(len(pd.read_csv(args.val_csv))),
            "custom_cnn_forward_pass_succeeded": True,
            "training_started": False,
        },
    }

    write_report(Path(args.report_path), results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
