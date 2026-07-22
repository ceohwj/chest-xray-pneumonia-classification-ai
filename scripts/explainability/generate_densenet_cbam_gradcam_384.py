"""Generate Grad-CAM images for the DenseNet121-CBAM 384px model.

This script is for portfolio explainability analysis only. The generated
visualizations are not clinical evidence and should not be used for diagnosis.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples for DenseNet121-CBAM 384px checkpoints.")
    parser.add_argument("--oof-csv", default="outputs/results(마지막)/oof_densenet121.csv")
    parser.add_argument("--image-dir", default="data/images")
    parser.add_argument("--checkpoint-dir", default="outputs/results(마지막)")
    parser.add_argument("--output-dir", default="outputs/gradcam/densenet121_384")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--cases-per-type", type=int, default=1)
    return parser.parse_args()


class CLAHETransform:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img_pil)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
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
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        return self.sa(x)


class DenseNet121CBAM(nn.Module):
    def __init__(self) -> None:
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


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        prob = torch.sigmoid(logits).reshape(-1)[0]
        logits.reshape(-1)[0].backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        cam_np -= cam_np.min()
        max_value = cam_np.max()
        if max_value > 0:
            cam_np /= max_value
        return cam_np, float(prob.detach().cpu().item())

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(image_size: int) -> Any:
    return transforms.Compose(
        [
            CLAHETransform(),
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def resolve_image_path(image_dir: Path, file_name: str) -> Path:
    raw = Path(file_name)
    candidates = [
        image_dir / raw,
        image_dir / "train" / raw.name,
        image_dir / "test" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for file_name={file_name}")


def load_model(checkpoint_dir: Path, fold: int, device: torch.device) -> DenseNet121CBAM:
    checkpoint_path = checkpoint_dir / f"best_densenet121_384_fold{fold}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model = DenseNet121CBAM().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def overlay_cam(image_path: Path, cam: np.ndarray, image_size: int) -> np.ndarray:
    original = Image.open(image_path).convert("RGB")
    original = original.resize((int(image_size * 1.1), int(image_size * 1.1)))
    left = (original.width - image_size) // 2
    top = (original.height - image_size) // 2
    original = original.crop((left, top, left + image_size, top + image_size))
    rgb = np.array(original)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * rgb + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    return np.concatenate([rgb, overlay], axis=1)


def choose_cases(oof_df: pd.DataFrame, threshold: float, cases_per_type: int) -> pd.DataFrame:
    df = oof_df.copy()
    df["pred"] = (df["prob"] > threshold).astype(int)
    case_frames = []

    correct = df[(df["label"] == 1) & (df["pred"] == 1)].sort_values("prob", ascending=False).head(cases_per_type).copy()
    correct["case_type"] = "correct_pneumonia"
    case_frames.append(correct)

    fn = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("prob", ascending=False).head(cases_per_type).copy()
    fn["case_type"] = "fn"
    case_frames.append(fn)

    fp = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("prob", ascending=False).head(cases_per_type).copy()
    fp["case_type"] = "fp"
    case_frames.append(fp)

    selected = pd.concat(case_frames, ignore_index=True)
    if selected.empty:
        raise RuntimeError("No Grad-CAM cases selected. Check OOF CSV and threshold.")
    return selected


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_df = pd.read_csv(args.oof_csv)
    selected = choose_cases(oof_df, args.threshold, args.cases_per_type)
    selected.to_csv(output_dir / "selected_cases.csv", index=False)

    transform = build_transform(args.image_size)
    model_cache: dict[int, DenseNet121CBAM] = {}

    rows = []
    for row in selected.itertuples(index=False):
        fold = int(row.fold)
        model = model_cache.get(fold)
        if model is None:
            model = load_model(Path(args.checkpoint_dir), fold, device)
            model_cache[fold] = model

        image_path = resolve_image_path(Path(args.image_dir), str(row.file_name))
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        gradcam = GradCAM(model, model.features.denseblock4)
        cam, regenerated_prob = gradcam(image_tensor)
        gradcam.close()

        rendered = overlay_cam(image_path, cam, args.image_size)
        case_dir = output_dir / str(row.case_type)
        case_dir.mkdir(parents=True, exist_ok=True)
        out_path = case_dir / f"{Path(row.file_name).stem}_fold{fold}_gradcam.png"
        Image.fromarray(rendered).save(out_path)
        rows.append(
            {
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
        print(f"Saved {row.case_type}: {out_path}")

    pd.DataFrame(rows).to_csv(output_dir / "gradcam_index.csv", index=False)
    print(f"Grad-CAM index saved to {output_dir / 'gradcam_index.csv'}")


if __name__ == "__main__":
    main()
