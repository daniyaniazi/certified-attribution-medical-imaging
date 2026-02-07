#!/usr/bin/env python
"""Better test of ranking-based sparsification fix."""

import numpy as np
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.certify.smoothing import RandomizedSmoothingAttributor

# Create synthetic sparse heatmap
heatmap = np.zeros((256, 256))
h, w = 256, 256
cell_h, cell_w = 128, 128

# Cell 0: significant values 0.5-1.0 in top-left, rest in 0.04-0.08
np.random.seed(42)
cell_0 = np.random.uniform(0.04, 0.08, size=(cell_h, cell_w))
cell_0[0:50, 0:50] = np.random.uniform(0.5, 1.0, size=(50, 50))  # Bright region (2500 pixels)
heatmap[0:cell_h, 0:cell_w] = cell_0

# Cells 1,2,3: all near-zero or slightly negative (which should not affect sorting)
heatmap[0:cell_h, cell_w:] = np.random.uniform(0.0, 0.01, size=(cell_h, cell_w))
heatmap[cell_h:, 0:cell_w] = np.random.uniform(0.0, 0.01, size=(cell_h, cell_w))
heatmap[cell_h:, cell_w:] = np.random.uniform(0.0, 0.01, size=(cell_h, cell_w))

print("Synthetic sparse heatmap (mimicking realistic DiFull output):")
print(f"  Cell 0 bright region (0:50, 0:50): mean={heatmap[0:50, 0:50].mean():.3f}, max={heatmap[0:50, 0:50].max():.3f}")
print(f"  Cell 0 rest: mean={heatmap[50:cell_h, 50:cell_w].mean():.6f}")
print(f"  Cell 1: mean={heatmap[0:cell_h, cell_w:].mean():.6f}")
print(f"  Cell 2: mean={heatmap[cell_h:, 0:cell_w].mean():.6f}")
print(f"  Cell 3: mean={heatmap[cell_h:, cell_w:].mean():.6f}")
print(f"  Total nonzero pixels: {(heatmap > 1e-6).sum()}")

# Test ranking-based sparsification
smoother = RandomizedSmoothingAttributor(None, None, device='cpu')
mask = smoother._sparsify_topk(heatmap, 50)

print(f"\n--- RANKING-BASED SPARSIFICATION (K=50%) ---")
print(f"Total pixels marked as '1': {mask.sum():.0f} / {mask.size} = {100*mask.mean():.1f}%")

for cell_id in range(4):
    r = (cell_id // 2) * cell_h
    c = (cell_id % 2) * cell_w
    m = mask[r:r+cell_h, c:c+cell_w]
    print(f"  Cell {cell_id}: {m.sum():.0f} / {cell_h*cell_w} = {100*m.mean():.1f}%")

# Expected: top 50% by value should come almost entirely from Cell 0's bright region
top_k_pixels = np.sum(heatmap > np.percentile(heatmap, 50))
print(f"\nExpected top 50%: ~{32768} pixels")
print(f"Actual top 50%: {mask.sum():.0f} pixels")

if mask[0:50, 0:50].sum() > 2000:  # At least 80% of bright region
    print("\n✓ PASS: Ranking correctly identifies Cell 0 bright region as top K%")
else:
    print("\n✗ FAIL: Ranking not working correctly")
