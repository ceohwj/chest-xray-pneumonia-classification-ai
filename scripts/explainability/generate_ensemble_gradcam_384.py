"""Generate Grad-CAM examples for all 384px ensemble backbones.

Models:
- DenseNet121-CBAM
- EfficientNet-B3
- ConvNeXt-Tiny

Research/education portfolio project only. These visualizations are not
clinical evidence and should not be used for diagnosis.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from .generate_densenet_cbam_gradcam_384 import (
    CBAM,
    DenseNet121CBAM,
    GradCAM,
    build_transform,
    choose_cases,
    overlay_cam,
    resolve_image_path,
    select_device,
)


MODEL_CONFIG = {
    "densenet121": {
        "display_name": "DenseNet121-CBAM",
        "oof_csv": "outputs/results(마지막)/oof_densenet121.csv",
        "checkpoint_pattern": "best_densenet121_384_fold{fold}.pt",
    },
    "efficientnet_b3": {
        "display_name": "EfficientNet-B3",
        "oof_csv": "outputs/results(마지막)/oof_efficientnet_b3.csv",
        "checkpoint_pattern": "best_efficientnet_b3_384_fold{fold}.pt",
    },
    "convnext_tiny": {
        "display_name": "ConvNeXt-Tiny",
        "oof_csv": "outputs/results(마지막)/oof_convnext_tiny.csv",
        "checkpoint_pattern": "best_convnext_tiny_384_fold{fold}.pt",
    },
}


class ConvNeXtTinyCBAM(nn.Module):
    def __init__(self) -> None:
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
    def __init__(self) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples for every 384px ensemble model.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIG), choices=sorted(MODEL_CONFIG))
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--checkpoint-dir", default="outputs/results(마지막)")
    parser.add_argument("--output-dir", default="outputs/gradcam/ensemble_384")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--cases-per-type", type=int, default=1)
    return parser.parse_args()


def build_model(model_name: str) -> nn.Module:
    if model_name == "densenet121":
        return DenseNet121CBAM()
    if model_name == "efficientnet_b3":
        return EfficientNetB3CBAM()
    if model_name == "convnext_tiny":
        return ConvNeXtTinyCBAM()
    raise ValueError(f"Unsupported model: {model_name}")


def target_layer(model_name: str, model: nn.Module) -> nn.Module:
    if model_name == "densenet121":
        return model.features.denseblock4
    if model_name == "efficientnet_b3":
        return model.cbam
    if model_name == "convnext_tiny":
        return model.cbam
    raise ValueError(f"Unsupported model: {model_name}")


def load_model(model_name: str, checkpoint_dir: Path, fold: int, device: torch.device) -> nn.Module:
    config = MODEL_CONFIG[model_name]
    checkpoint_path = checkpoint_dir / config["checkpoint_pattern"].format(fold=fold)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model = build_model(model_name).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_for_model(model_name: str, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    config = MODEL_CONFIG[model_name]
    model_output_dir = Path(args.output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.read_csv(config["oof_csv"])
    selected = choose_cases(oof_df, args.threshold, args.cases_per_type)
    selected.to_csv(model_output_dir / "selected_cases.csv", index=False)

    transform = build_transform(args.image_size)
    model_cache: dict[int, nn.Module] = {}
    rows = []

    for row in selected.itertuples(index=False):
        fold = int(row.fold)
        model = model_cache.get(fold)
        if model is None:
            model = load_model(model_name, Path(args.checkpoint_dir), fold, device)
            model_cache[fold] = model

        image_path = resolve_image_path(Path(args.image_dir), str(row.file_name))
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        gradcam = GradCAM(model, target_layer(model_name, model))
        cam, regenerated_prob = gradcam(image_tensor)
        gradcam.close()

        rendered = overlay_cam(image_path, cam, args.image_size)
        case_dir = model_output_dir / str(row.case_type)
        case_dir.mkdir(parents=True, exist_ok=True)
        out_path = case_dir / f"{Path(row.file_name).stem}_fold{fold}_gradcam.png"
        Image.fromarray(rendered).save(out_path)

        rows.append(
            {
                "model": model_name,
                "display_name": config["display_name"],
                "case_type": row.case_type,
                "file_name": row.file_name,
                "label": int(row.label),
                "fold": fold,
                "oof_prob": float(row.prob),
                "regenerated_prob": regenerated_prob,
                "pred": int(row.pred),
                "gradcam_path": str(out_path),
            }
        )
        print(f"Saved {config['display_name']} {row.case_type}: {out_path}", flush=True)

    index_df = pd.DataFrame(rows)
    index_df.to_csv(model_output_dir / "gradcam_index.csv", index=False)
    return index_df


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model_name in args.models:
        all_rows.append(generate_for_model(model_name, args, device))

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(Path(args.output_dir) / "gradcam_index.csv", index=False)
    print(f"Combined Grad-CAM index saved to {Path(args.output_dir) / 'gradcam_index.csv'}")


if __name__ == "__main__":
    main()
