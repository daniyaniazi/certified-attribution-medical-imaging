#!/usr/bin/env python
"""Simple test of the sparsification fix without loading dataset."""

import numpy as np
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.certify.smoothing import RandomizedSmoothingAttributor

# Create a synthetic sparse heatmap similar to DiFull output
# Cell 0 (top-left): concentrated values 0.05-1.0
# Cells 1,2,3: all zeros
heatmap = np.zeros((256, 256))
h, w = 256, 256
cell_h, cell_w = 128, 128

# Cell 0: sparse values concentrated in top-left
cell_0 = np.random.uniform(0.04, 0.08, size=(cell_h, cell_w))
cell_0[0:50, 0:50] = np.random.uniform(0.5, 1.0, size=(50, 50))  # Bright region
heatmap[0:cell_h, 0:cell_w] = cell_0

# Cells 1,2,3: all zeros (no attribution)
# Already set by initialization

print("Synthetic sparse heatmap (mimicking DiFull output):")
print(f"  Cell 0: mean={heatmap[0:cell_h, 0:cell_w].mean():.6f}, "
      f"max={heatmap[0:cell_h, 0:cell_w].max():.6f}")
print(f"  Cell 1: mean={heatmap[0:cell_h, cell_w:].mean():.6f}")
print(f"  Cell 2: mean={heatmap[cell_h:, 0:cell_w].mean():.6f}")
print(f"  Cell 3: mean={heatmap[cell_h:, cell_w:].mean():.6f}")

# Create a dummy attributor to test sparsification
smoother = RandomizedSmoothingAttributor(None, None, device='cpu')

print("\n--- OLD METHOD (broken): percentile of ALL pixels ---")
flat = heatmap.flatten()
threshold_old = np.percentile(flat, 100 - 50)  # Top 50%
print(f"Old threshold: {threshold_old:.8f}")
mask_old = (heatmap >= threshold_old).astype(np.float32)
print(f"Pixels marked as 1: {mask_old.sum():.0f} / {mask_old.size} = {100*mask_old.mean():.1f}%")
for cell_id in range(4):
    r = (cell_id // 2) * cell_h
    c = (cell_id % 2) * cell_w
    m = mask_old[r:r+cell_h, c:c+cell_w].sum()
    print(f"  Cell {cell_id}: {m:.0f} / {cell_h*cell_w} = {100*m/(cell_h*cell_w):.1f}%")

print("\n--- NEW METHOD (fixed): percentile of NON-ZERO pixels ---")
mask_new = smoother._sparsify_topk(heatmap, 50)
print(f"Pixels marked as 1: {mask_new.sum():.0f} / {mask_new.size} = {100*mask_new.mean():.1f}%")
for cell_id in range(4):
    r = (cell_id // 2) * cell_h
    c = (cell_id % 2) * cell_w
    m = mask_new[r:r+cell_h, c:c+cell_w].sum()
    print(f"  Cell {cell_id}: {m:.0f} / {cell_h*cell_w} = {100*m/(cell_h*cell_w):.1f}%")

if mask_new[0, 0] > 0.5 and mask_new[cell_h, cell_h] < 0.5:
    print("\n✓ PASS: Sparsification correctly localizes to Cell 0!")
else:
    print("\n✗ FAIL: Sparsification did not localize properly")
