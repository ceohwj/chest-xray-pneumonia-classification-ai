"""Lightweight image transforms that do not require torchvision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class ResizeToTensor:
    """Resize a PIL image and convert it to a float tensor in [0, 1]."""

    image_size: int = 224

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
