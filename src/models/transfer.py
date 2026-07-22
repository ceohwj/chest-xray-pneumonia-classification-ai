"""Transfer-learning model factory for later experiments."""

from __future__ import annotations

from torch import nn


def create_transfer_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.4,
) -> nn.Module:
    """Create a torchvision transfer model when torchvision is installed."""
    try:
        import torchvision.models as models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchvision is required for transfer models. Install torchvision before running transfer-learning experiments."
        ) from exc

    name = model_name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in _classifier_parameters(model):
            param.requires_grad = True
    return model


def _classifier_parameters(model: nn.Module):
    if hasattr(model, "fc"):
        return model.fc.parameters()
    if hasattr(model, "classifier"):
        return model.classifier.parameters()
    return []
