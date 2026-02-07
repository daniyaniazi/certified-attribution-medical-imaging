#!/usr/bin/env python
"""Debug script to verify the sparsification fix."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.grid_dataset import GridDataset
from src.models.grid_multihead import GridMultiHead
from src.certify.smoothing import RandomizedSmoothingAttributor
from src.xai.attribution_unified import IntegratedGradientsUnified

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# Load grid dataset
grid_ds = GridDataset(ROOT / "data" / "processed" / "isic" / "grid_2x2.pt")
sample = grid_ds[0]
image = sample["image"].unsqueeze(0).to(device)
target_class = int(sample["target_class"].item())
head_id = int(sample["target_head"].item())

print(f"Image shape: {image.shape}")
print(f"Target class: {target_class}, Head ID: {head_id}\n")

# Load GridMultiHead model
model = GridMultiHead(
    "resnet18",
    num_classes=8,
    num_heads=4,
    pretrained=False,
    scale=2,
)
model.to(device)
model.eval()

# DiFull wrapper
class DiFull_Wrapper(nn.Module):
    def __init__(self, grid_model: GridMultiHead, head_id: int):
        super().__init__()
        self.grid_model = grid_model
        self.head_id = head_id

    def forward(self, x: torch.Tensor):
        return self.grid_model(x, head_id=self.head_id)

wrapper = DiFull_Wrapper(model, head_id)

# Get attribution for one sample
print("Computing attribution with IG...")
ig = IntegratedGradientsUnified(wrapper, device)
heatmap = ig.attribute(image, target_class=target_class)
print(f"Heatmap shape: {heatmap.shape}")
print(f"Heatmap min={heatmap.min():.6f}, max={heatmap.max():.6f}, mean={heatmap.mean():.6f}")

# Check per-cell values
h, w = heatmap.shape
cell_h, cell_w = h // 2, w // 2
for cell_id in range(4):
    r = (cell_id // 2) * cell_h
    c = (cell_id % 2) * cell_w
    cell_hmap = heatmap[r:r+cell_h, c:c+cell_w]
    print(f"Cell {cell_id}: mean={cell_hmap.mean():.6f}, max={cell_hmap.max():.6f}, "
          f"nonzero%={(cell_hmap > 1e-6).sum().item() * 100 / cell_hmap.numel():.1f}%")

print("\n--- Testing sparsification with original (broken) method ---")
flat = heatmap.flatten().numpy()
threshold_old = np.percentile(flat, 50)  # Top 50%
print(f"Old threshold (50th percentile of all): {threshold_old:.6f}")
mask_old = (heatmap.numpy() >= threshold_old).astype(np.float32)
print(f"Old mask: {mask_old.sum():.0f} / {mask_old.size} pixels marked as 1 ({100*mask_old.mean():.1f}%)")

print("\n--- Testing sparsification with fixed method ---")
non_zero = flat[flat > 1e-6]
if len(non_zero) > 0:
    threshold_new = np.percentile(non_zero, 50)  # Top 50% of NON-ZERO
    print(f"New threshold (50th percentile of non-zero): {threshold_new:.6f}")
else:
    threshold_new = 0.0
    print(f"New threshold: no non-zero values")

mask_new = (heatmap.numpy() >= threshold_new).astype(np.float32)
print(f"New mask: {mask_new.sum():.0f} / {mask_new.size} pixels marked as 1 ({100*mask_new.mean():.1f}%)")

print("\n--- Per-cell sparsified mask ---")
for cell_id in range(4):
    r = (cell_id // 2) * cell_h
    c = (cell_id % 2) * cell_w
    cell_mask = mask_new[r:r+cell_h, c:c+cell_w]
    print(f"Cell {cell_id}: {cell_mask.sum():.0f} / {cell_mask.size} pixels marked as 1 ({100*cell_mask.mean():.1f}%)")

print("\n✓ Sparsification fix should now preserve per-cell localization!")
