#!/usr/bin/env python
"""Debug: Check what the grid dataset actually contains."""

import sys
from pathlib import Path
import torch

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.grid_dataset import GridDataset

# Load the grid dataset
grid_pt = Path("data/raw/grid/isic/val/grid.pt")
if not grid_pt.exists():
    print(f"❌ Grid file not found: {grid_pt}")
    sys.exit(1)

print(f"Loading grid from: {grid_pt}")
grid_ds = GridDataset(grid_pt)

print(f"\nDataset info:")
print(f"  Total samples: {len(grid_ds)}")
print(f"  Grid scale: {grid_ds.scale}x{grid_ds.scale}")
print(f"  Target cell: {grid_ds.target_cell}")

# Get first sample
sample = grid_ds[0]
image = sample["image"]
target_class = sample["target_class"]
target_head = sample["target_head"]

print(f"\nFirst sample:")
print(f"  Image shape: {image.shape}")
print(f"  Target class: {target_class}")
print(f"  Target head (cell): {target_head}")

# Check if all cells are the same by comparing pixel values
print(f"\nAnalyzing grid cells:")
cell_H = image.shape[1] // grid_ds.scale
cell_W = image.shape[2] // grid_ds.scale

for cell_idx in range(grid_ds.scale * grid_ds.scale):
    row = cell_idx // grid_ds.scale
    col = cell_idx % grid_ds.scale
    y0, y1 = row * cell_H, (row + 1) * cell_H
    x0, x1 = col * cell_W, (col + 1) * cell_W
    
    cell = image[:, y0:y1, x0:x1]
    cell_mean = cell.mean().item()
    cell_std = cell.std().item()
    
    marker = " ← TARGET" if cell_idx == target_head else ""
    print(f"  Cell {cell_idx} [{y0}:{y1}, {x0}:{x1}]: mean={cell_mean:.4f}, std={cell_std:.4f}{marker}")

# Check if cells are identical
cell_0 = image[:, 0:cell_H, 0:cell_W]
cell_1 = image[:, 0:cell_H, cell_W:2*cell_W]
cell_2 = image[:, cell_H:2*cell_H, 0:cell_W]
cell_3 = image[:, cell_H:2*cell_H, cell_W:2*cell_W]

print(f"\nCell similarity check:")
print(f"  Cell 0 vs Cell 1: max_diff={(cell_0 - cell_1).abs().max().item():.6f}")
print(f"  Cell 0 vs Cell 2: max_diff={(cell_0 - cell_2).abs().max().item():.6f}")
print(f"  Cell 0 vs Cell 3: max_diff={(cell_0 - cell_3).abs().max().item():.6f}")

if (cell_0 - cell_1).abs().max() < 0.01 and (cell_0 - cell_2).abs().max() < 0.01:
    print("\n❌ WARNING: All cells appear to be the SAME image!")
    print("This would explain why all cells are getting certified.")
else:
    print("\n✅ Cells are DIFFERENT images (as expected)")
