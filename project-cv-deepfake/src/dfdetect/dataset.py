"""Torch Dataset wrapper — the (path, label) list logic lives in data.py (pure, no
torch) so it's independently testable; this module adds the image-loading/tensor path.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceDataset(Dataset):
    def __init__(self, items: list[tuple], img_size: int, augment: bool = False):
        self.items = items
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            arr = arr[:, ::-1, :].copy()  # horizontal flip
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
        return tensor, torch.tensor(label, dtype=torch.float32)
