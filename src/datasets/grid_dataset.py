"""Grid dataset loader for composite n×n images with per-cell labels.

Assumes payload saved by src.certify.utils.generate_grid_dataset:
- images: torch.Tensor [N, C, H*scale, W*scale]
- cell_classes: [N, scale*scale]
- target_classes: [N]
- target_cell: int
- scale: int
- image_size: (H, W)
- class_names: optional list
"""
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset


class GridDataset(Dataset):
    def __init__(self, payload_path: Path):
        super().__init__()
        payload = torch.load(payload_path)
        self.images = payload["images"]
        self.cell_classes = payload["cell_classes"]
        self.target_classes = payload["target_classes"]
        self.target_cell = int(payload.get("target_cell", 0))
        self.scale = int(payload.get("scale", 2))
        self.image_size = tuple(payload.get("image_size", (224, 224)))
        self.class_names = payload.get("class_names", None)
        self.meta = {
            "dataset": payload.get("dataset", "grid"),
            "split": payload.get("split", None),
            "num_classes": int(payload.get("num_classes", -1)),
        }

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.images[idx]
        cell_classes = self.cell_classes[idx]
        target_class = int(self.target_classes[idx])
        return {
            "image": image,
            "cell_classes": cell_classes,
            "target_class": target_class,
            "target_head": int(self.target_cell),
            "meta": {
                "scale": self.scale,
                "image_size": self.image_size,
                "target_cell": self.target_cell,
                "class_names": self.class_names,
            },
        }

    def build_target_mask(self) -> Dict[int, torch.Tensor]:
        """Return binary masks (per sample) for the target cell region."""
        masks = {}
        h, w = self.image_size
        full_h, full_w = h * self.scale, w * self.scale
        row = self.target_cell // self.scale
        col = self.target_cell % self.scale
        y0, x0 = row * h, col * w
        for idx in range(len(self)):
            mask = torch.zeros((full_h, full_w), dtype=torch.uint8)
            mask[y0:y0 + h, x0:x0 + w] = 1
            masks[idx] = mask
        return masks
