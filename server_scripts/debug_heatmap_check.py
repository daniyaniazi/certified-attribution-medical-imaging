#!/usr/bin/env python
"""Quick check: is the attribution heatmap actually localizing to target cell?"""
import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.grid_dataset import GridDataset
from src.models.grid_multihead import GridMultiHead
from src.xai.attribution_unified import IntegratedGradientsUnified
from torch.utils.data import DataLoader

# Load model and data
try:
    grid_ds = GridDataset(Path("data/raw/grid/isic/val/grid.pt"))
except:
    print("Grid dataset not found, trying alternative path...")
    grid_ds = GridDataset(Path("data/processed/isic/grid.pt"))

loader = DataLoader(grid_ds, batch_size=1, shuffle=False)
device = "cpu"  # Force CPU for speed
print(f"Using device: {device}")

print("Loading model...")
model = GridMultiHead("resnet18", num_classes=8, num_heads=4, pretrained=True, scale=2)
model.to(device).eval()

# Get one sample
print("Getting sample...")
sample = next(iter(loader))
image = sample["image"].to(device)
head_id = int(sample["target_head"].item())
target_class = int(sample["target_class"].item())

print(f"\nImage shape: {image.shape}")
print(f"Target head (cell): {head_id}")
print(f"Target class: {target_class}")

# Test DiFull_Wrapper
class TestWrapper(torch.nn.Module):
    def __init__(self, model, head_id):
        super().__init__()
        self.model = model
        self.head_id = head_id
    
    def forward(self, x):
        return self.model(x, head_id=self.head_id)

wrapper = TestWrapper(model, head_id)

# Compute IG attribution
attr_method = IntegratedGradientsUnified(wrapper, device)
with torch.enable_grad():
    heatmap = attr_method.attribute(image, target_class=target_class)

# Analyze heatmap
print(f"\nHeatmap shape: {heatmap.shape}")
print(f"Heatmap min: {heatmap.min():.6f}, max: {heatmap.max():.6f}, mean: {heatmap.mean():.6f}")
print(f"Heatmap std: {heatmap.std():.6f}")

# Split into 4 cells
h, w = heatmap.shape
cell_h, cell_w = h // 2, w // 2

cell_0 = heatmap[0:cell_h, 0:cell_w]  # top-left
cell_1 = heatmap[0:cell_h, cell_w:w]   # top-right
cell_2 = heatmap[cell_h:h, 0:cell_w]   # bottom-left
cell_3 = heatmap[cell_h:h, cell_w:w]   # bottom-right

cells = [cell_0, cell_1, cell_2, cell_3]
print(f"\nAttribution per cell:")
for i, cell in enumerate(cells):
    cell_np = np.array(cell)
    pct_nonzero = (cell_np > 1e-6).sum() / cell_np.size * 100
    print(f"  Cell {i} (head_id={head_id}): mean={cell_np.mean():.6f}, max={cell_np.max():.6f}, "
          f"nonzero%={pct_nonzero:.1f}%")

print(f"\nExpected: Cell {head_id} should have high attribution, others near-zero")
print(f"\nDiFull working correctly if:")
print(f"  - Cell {head_id} has mean >> other cells")
print(f"  - Other cells have mean ≈ 0 or very small")
