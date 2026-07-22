"""Load and smoke-test every checkpoint in the packaged reference ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes: int):
        super().__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sa(self.ca(x))


class DenseNet121CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=None)
        self.features = base.features
        self.cbam = CBAM(1024)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cbam(F.relu(self.features(x), inplace=True))
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(features, (1, 1)), 1))


class ConvNeXtTinyCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.cbam = CBAM(768)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cbam(self.features(x))
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(features, (1, 1)), 1))


class EfficientNetB3CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_b3(weights=None)
        self.features = base.features
        self.cbam = CBAM(1536)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1536, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cbam(self.features(x))
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(features, (1, 1)), 1))


MODEL_FACTORIES = {
    "densenet121": DenseNet121CBAM,
    "convnext_tiny": ConvNeXtTinyCBAM,
    "efficientnet_b3": EfficientNetB3CBAM,
}


class CLAHETransform:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image: Image.Image) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        enhanced = self.clahe.apply(lightness)
        result = cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)


class PadToSquare:
    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(image, ((side - width) // 2, (side - height) // 2))
        return padded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=Path("outputs/best_pb_ensemble"))
    parser.add_argument("--image", type=Path, default=Path("data/images/test/test_0001.png"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/best_pb_ensemble/verification_result.json"),
    )
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    return parser.parse_args()


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            CLAHETransform(),
            PadToSquare(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.package_dir / "manifest.json").read_text(encoding="utf-8"))
    image_size = int(manifest["image_size"])
    threshold = float(manifest["threshold"])
    device = torch.device(args.device)
    image_tensor = build_transform(image_size)(Image.open(args.image)).unsqueeze(0).to(device)

    model_probabilities: dict[str, float] = {}
    checkpoint_results: list[dict[str, object]] = []
    with torch.inference_mode():
        for model_name, relative_paths in manifest["checkpoints"].items():
            fold_probabilities = []
            for fold, relative_path in enumerate(relative_paths):
                checkpoint_path = args.package_dir / relative_path
                model = MODEL_FACTORIES[model_name]().to(device)
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                incompatible = model.load_state_dict(state_dict, strict=True)
                model.eval()
                probability = (
                    torch.sigmoid(model(image_tensor))
                    + torch.sigmoid(model(torch.flip(image_tensor, dims=[3])))
                ).div(2.0).item()
                fold_probabilities.append(probability)
                checkpoint_results.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "checkpoint": relative_path,
                        "missing_keys": list(incompatible.missing_keys),
                        "unexpected_keys": list(incompatible.unexpected_keys),
                        "probability_pneumonia": probability,
                    }
                )
                del model, state_dict
            model_probabilities[model_name] = float(np.mean(fold_probabilities))

    ensemble_probability = sum(
        float(manifest["ensemble_weights"][name]) * probability
        for name, probability in model_probabilities.items()
    )
    predicted_label = int(ensemble_probability >= threshold)

    expected_label = None
    submission_path = args.package_dir / manifest["submission"]
    with submission_path.open(encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        file_index = header.index("file_name")
        label_index = header.index("label")
        for line in handle:
            row = line.strip().split(",")
            if row[file_index] == args.image.name:
                expected_label = int(row[label_index])
                break

    result = {
        "status": "PASS" if expected_label == predicted_label else "FAIL",
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "device": str(device),
        "image": str(args.image),
        "image_tensor_shape": list(image_tensor.shape),
        "loaded_checkpoint_count": len(checkpoint_results),
        "strict_load_all_passed": all(
            not row["missing_keys"] and not row["unexpected_keys"] for row in checkpoint_results
        ),
        "model_probabilities": model_probabilities,
        "ensemble_probability_pneumonia": ensemble_probability,
        "threshold": threshold,
        "predicted_label": predicted_label,
        "submission_label": expected_label,
        "prediction_matches_submission": expected_label == predicted_label,
        "checkpoint_results": checkpoint_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "checkpoint_results"}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit("Verification prediction does not match the packaged submission.")


if __name__ == "__main__":
    main()
