#!/usr/bin/env python
"""Comprehensive test of certification pipeline with sparsification fix."""

import numpy as np
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.certify.smoothing import RandomizedSmoothingAttributor

# Simulate multiple noisy samples of the sparse heatmap
# This models the randomized smoothing process
print("=== Simulating Randomized Smoothing Certification ===\n")

# Create synthetic heatmaps for 10 noisy samples
# Cell 0 always concentrated, Cells 1-3 always near zero
num_samples = 10
heatmaps = []
for sample_id in range(num_samples):
    hmap = np.zeros((256, 256))
    h, w = 256, 256
    cell_h, cell_w = 128, 128
    
    # Cell 0: consistent concentrated values with noise
    cell_0 = np.random.uniform(0.04, 0.08, size=(cell_h, cell_w))
    cell_0[0:50, 0:50] += np.random.normal(0, 0.1, size=(50, 50))  # Add noise
    cell_0 = np.clip(cell_0, 0, 1)
    hmap[0:cell_h, 0:cell_w] = cell_0
    
    # Cells 1,2,3: mostly zeros with occasional small noise
    for cell_id in [1, 2, 3]:
        r = (cell_id // 2) * cell_h
        c = (cell_id % 2) * cell_w
        hmap[r:r+cell_h, c:c+cell_w] = np.random.uniform(0, 0.01, size=(cell_h, cell_w))
    
    heatmaps.append(hmap)

# Sparsify each heatmap
smoother = RandomizedSmoothingAttributor(None, None, device='cpu')
masks = []
for hmap in heatmaps:
    mask = smoother._sparsify_topk(hmap, 50)
    masks.append(mask)

print("Per-sample sparsified masks (K=50%):")
for i, mask in enumerate(masks[:3]):
    cell_0_sum = mask[0:128, 0:128].sum()
    other_sum = (mask[0:128, 128:].sum() + mask[128:, 0:128].sum() + mask[128:, 128:].sum())
    print(f"  Sample {i}: Cell0={cell_0_sum:.0f}/16384 ({100*cell_0_sum/16384:.1f}%), "
          f"Others={other_sum:.0f}/49152 ({100*other_sum/49152:.2f}%)")

# Aggregate via majority voting (as in certification)
print("\nMajority voting aggregation:")
p_1 = np.mean(np.array(masks), axis=0)  # Mean of all masks = probability of "1"
p_0 = 1 - p_1

print(f"  p_1[Cell 0]: mean={p_1[50, 50]:.3f} (concentrated region)")
print(f"  p_1[Cell 1]: mean={p_1[200, 200]:.3f} (other cell)")
print(f"  p_0[Cell 0]: mean={p_0[50, 50]:.3f}")
print(f"  p_0[Cell 1]: mean={p_0[200, 200]:.3f}")

# Certification with threshold tau=0.75
tau = 0.75
cert_1_cell0 = (p_1[0:128, 0:128] >= tau).sum()
cert_1_cell1 = (p_1[128:, 128:] >= tau).sum()
cert_0_cell0 = (p_0[0:128, 0:128] >= tau).sum()
cert_0_cell1 = (p_0[128:, 128:] >= tau).sum()

print(f"\nCertification with τ={tau}:")
print(f"  Cell 0: {cert_1_cell0:.0f} certified as '1', {cert_0_cell0:.0f} certified as '0'")
print(f"  Cell 1: {cert_1_cell1:.0f} certified as '1', {cert_0_cell1:.0f} certified as '0'")

if cert_1_cell0 > 5000 and cert_0_cell1 > 5000:
    print("\n✓ SUCCESS: Certification correctly distinguishes Cell 0 from other cells!")
    print("  Cell 0 heavily certified as '1' (top cell)")
    print("  Cell 1 heavily certified as '0' (not top cell)")
else:
    print("\n✗ ISSUE: Certification not localizing properly")
    print(f"  Cell 0 certified-1: {cert_1_cell0:.0f} (expected > 5000)")
    print(f"  Cell 1 certified-0: {cert_0_cell1:.0f} (expected > 5000)")
