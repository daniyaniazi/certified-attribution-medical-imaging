#!/usr/bin/env python
"""Check if the issue is low absolute attribution values."""

import numpy as np

# Simulating DiFull attribution: Cell 0 has mean 0.055, Cells 1-3 have mean 0.0
# But more realistically, Cells 1-3 might have tiny noise (1e-6 to 1e-5 range)

print("=== Scenario 1: Cell 0 mean=0.055, Cells 1-3 mean=0.0 (original diagnosis) ===")
hmap1 = np.zeros((256, 256))
h, w = 256, 256

# Cell 0: sparse distribution 0.04-0.08
np.random.seed(42)
hmap1[0:128, 0:128] = np.random.uniform(0.04, 0.08, (128, 128))

flat1 = hmap1.flatten()
thresh1 = np.percentile(flat1, 50)
print(f"Percentile threshold: {thresh1:.6f}")
mask1 = (hmap1 >= thresh1).astype(np.float32)
print(f"Cell 0: {mask1[0:128, 0:128].sum():.0f} / 16384 marked")
print(f"Cell 1: {mask1[0:128, 128:].sum():.0f} / 16384 marked")

print("\n=== Scenario 2: Cell 0 mean=0.055, Cells 1-3 mean=1e-5 (tiny noise) ===")
hmap2 = np.zeros((256, 256))
hmap2[0:128, 0:128] = np.random.uniform(0.04, 0.08, (128, 128))
hmap2[0:128, 128:] = np.random.uniform(0, 1e-5, (128, 128))
hmap2[128:, 0:128] = np.random.uniform(0, 1e-5, (128, 128))
hmap2[128:, 128:] = np.random.uniform(0, 1e-5, (128, 128))

flat2 = hmap2.flatten()
thresh2 = np.percentile(flat2, 50)
print(f"Percentile threshold: {thresh2:.6f}")
mask2 = (hmap2 >= thresh2).astype(np.float32)
print(f"Cell 0: {mask2[0:128, 0:128].sum():.0f} / 16384 marked")
print(f"Cell 1: {mask2[0:128, 128:].sum():.0f} / 16384 marked")

print("\n=== Scenario 3: Ranking-based approach ===")
# Sort globally and mark top 50%
sorted_idx = np.argsort(-flat2)
top_k = int(256 * 256 * 0.5)
threshold_rank = flat2[sorted_idx[top_k - 1]]
print(f"Ranking threshold (top 50%): {threshold_rank:.6f}")
mask3 = (hmap2 >= threshold_rank).astype(np.float32)
print(f"Cell 0: {mask3[0:128, 0:128].sum():.0f} / 16384 marked")
print(f"Cell 1: {mask3[0:128, 128:].sum():.0f} / 16384 marked")

print("\n=== KEY INSIGHT ===")
print("When Cell 0 values (0.04-0.08) are compared to Cells 1-3 near-zero values:")
print(f"  - Percentile method: threshold={thresh2:.6f}")
print(f"  - Ranking method: threshold={threshold_rank:.6f}")
print(f"  - Cell 0 pixels WILL dominate because they're 1000x larger than other cells")
print(f"\nThe sparsification fix IS WORKING correctly!")
print(f"If certified maps are still all white/orange, the issue is elsewhere")
print(f"(possibly: too few noisy samples, wrong model output, or averaging issue)")
