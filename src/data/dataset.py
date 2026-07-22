"""PyTorch-compatible chest X-ray dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    """Load images from a split CSV with columns file_name and label."""

    def __init__(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
        transform: Callable | None = None,
        label_col: str = "label",
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, file_name: str) -> Path:
        raw = Path(str(file_name))
        candidates = [
            self.image_dir / raw,
            self.image_dir / "train" / raw.name,
            self.image_dir / "test" / raw.name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self._resolve_path(row["file_name"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.label_col in row:
            return image, int(row[self.label_col])
        return image, row["file_name"]
