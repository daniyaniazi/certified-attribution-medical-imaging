#!/usr/bin/env python
"""Check if heatmap normalization is the issue"""
import sys
from pathlib import Path
import torch
import numpy as np
from matplotlib import pyplot as plt

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.grid_dataset import GridDataset
from src.models.grid_multihead import GridMultiHead
from src.xai.attribution_unified import IntegratedGradientsUnified
from torch.utils.data import DataLoader

grid_ds = GridDataset(Path("data/raw/grid/isic/val/grid.pt"))
loader = DataLoader(grid_ds, batch_size=1, shuffle=False)
device = "cpu"

model = GridMultiHead("resnet18", num_classes=8, num_heads=4, pretrained=True, scale=2)
model.to(device).eval()

sample = next(iter(loader))
image = sample["image"].to(device)
head_id = int(sample["target_head"].item())
target_class = int(sample["target_class"].item())

class TestWrapper(torch.nn.Module):
    def __init__(self, model, head_id):
        super().__init__()
        self.model = model
        self.head_id = head_id
    
    def forward(self, x):
        return self.model(x, head_id=self.head_id)

wrapper = TestWrapper(model, head_id)
attr_method = IntegratedGradientsUnified(wrapper, device)

with torch.enable_grad():
    heatmap_raw = attr_method.attribute(image, target_class=target_class)

# Normalize to [0, 1]
heatmap_min = heatmap_raw.min()
heatmap_max = heatmap_raw.max()
heatmap_norm = (heatmap_raw - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)

print(f"Raw heatmap: min={heatmap_raw.min():.6f}, max={heatmap_raw.max():.6f}, mean={heatmap_raw.mean():.6f}")
print(f"Normalized:  min={heatmap_norm.min():.6f}, max={heatmap_norm.max():.6f}, mean={heatmap_norm.mean():.6f}")

# Check per-cell after normalization
h, w = heatmap_norm.shape
cell_h, cell_w = h // 2, w // 2

for i in range(4):
    row = i // 2
    col = i % 2
    y0, y1 = row * cell_h, (row + 1) * cell_h
    x0, x1 = col * cell_w, (col + 1) * cell_w
    cell = heatmap_norm[y0:y1, x0:x1]
    pct_nonzero = (cell > 1e-6).sum() / cell.size * 100
    print(f"Cell {i} (normalized): mean={cell.mean():.6f}, max={cell.max():.6f}, nonzero%={pct_nonzero:.1f}%")

# Test sparsification at K=50%
flat = heatmap_norm.flatten()
threshold = np.percentile(flat, 50)
print(f"\nSparsification at K=50%:")
print(f"Threshold value: {threshold:.6f}")

mask = (heatmap_norm >= threshold).astype(np.float32)
for i in range(4):
    row = i // 2
    col = i % 2
    y0, y1 = row * cell_h, (row + 1) * cell_h
    x0, x1 = col * cell_w, (col + 1) * cell_w
    cell_mask = mask[y0:y1, x0:x1]
    pct_in_top50 = cell_mask.mean() * 100
    print(f"Cell {i}: {pct_in_top50:.1f}% in top 50%")
