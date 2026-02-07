#!/usr/bin/env python
"""End-to-end test of certification with fixed sparsification."""

import numpy as np
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.certify.smoothing import RandomizedSmoothingAttributor

print("=== END-TO-END CERTIFICATION TEST WITH FIXED SPARSIFICATION ===\n")

# Simulate realistic DiFull attribution across 10 noisy samples
# Cell 0: concentrated 0.04-0.08 range (attribution to target cell)
# Cells 1-3: tiny noise 0.0-1e-5 range (no meaningful attribution)
np.random.seed(42)
num_samples = 10
masks_by_sample = []

for sample_id in range(num_samples):
    hmap = np.zeros((256, 256))
    
    # Cell 0: robust attribution across samples (0.04-0.08 range)
    hmap[0:128, 0:128] = np.random.uniform(0.04, 0.08, (128, 128))
    
    # Cells 1-3: tiny noise (should not affect top-K)
    hmap[0:128, 128:] = np.random.uniform(0, 1e-5, (128, 128))
    hmap[128:, 0:128] = np.random.uniform(0, 1e-5, (128, 128))
    hmap[128:, 128:] = np.random.uniform(0, 1e-5, (128, 128))
    
    # Sparsify with fixed ranking-based method
    smoother = RandomizedSmoothingAttributor(None, None, device='cpu')
    mask = smoother._sparsify_topk(hmap, 50)
    masks_by_sample.append(mask)
    
    if sample_id < 3:
        c0_pct = 100 * mask[0:128, 0:128].mean()
        c1_pct = 100 * mask[0:128, 128:].mean()
        print(f"Sample {sample_id}: Cell0={c0_pct:.1f}%, Cell1={c1_pct:.1f}%")

# Step 1: Aggregate via majority voting (Eq. 5 step 2)
masks_array = np.array(masks_by_sample)
p_1 = np.mean(masks_array, axis=0)  # Probability of "1"
p_0 = 1.0 - p_1

print(f"\nAfter majority voting aggregation:")
print(f"  p_1[Cell 0 center]: {p_1[64, 64]:.3f}")
print(f"  p_1[Cell 1 center]: {p_1[64, 192]:.3f}")
print(f"  p_0[Cell 0 center]: {p_0[64, 64]:.3f}")
print(f"  p_0[Cell 1 center]: {p_0[64, 192]:.3f}")

# Step 2: Certification with Clopper-Pearson bounds (simplified for testing)
# Using simple threshold: if p_1 >= 0.75, certify as "1"; if p_0 >= 0.75, certify as "0"
tau = 0.75
certified = np.full((256, 256), -1, dtype=np.int8)  # -1=abstain
certified[p_1 >= tau] = 1
certified[p_0 >= tau] = 0

print(f"\nCertification with τ={tau}:")
print(f"  Cell 0:")
c0_cert_1 = (certified[0:128, 0:128] == 1).sum()
c0_cert_0 = (certified[0:128, 0:128] == 0).sum()
c0_abstain = (certified[0:128, 0:128] == -1).sum()
print(f"    Certified '1': {c0_cert_1} / 16384")
print(f"    Certified '0': {c0_cert_0} / 16384")
print(f"    Abstained: {c0_abstain} / 16384")

print(f"  Cell 1:")
c1_cert_1 = (certified[0:128, 128:] == 1).sum()
c1_cert_0 = (certified[0:128, 128:] == 0).sum()
c1_abstain = (certified[0:128, 128:] == -1).sum()
print(f"    Certified '1': {c1_cert_1} / 16384")
print(f"    Certified '0': {c1_cert_0} / 16384")
print(f"    Abstained: {c1_abstain} / 16384")

# Validate results
print("\n=== VALIDATION ===")
if c0_cert_1 > 10000:
    print("✓ Cell 0 heavily certified as '1' (top attribution region)")
else:
    print(f"✗ Cell 0 not heavily certified as '1' ({c0_cert_1})")

if c1_cert_0 > 10000:
    print("✓ Cell 1 heavily certified as '0' (non-attribution region)")
else:
    print(f"✗ Cell 1 not heavily certified as '0' ({c1_cert_0})")

if c0_cert_1 > 10000 and c1_cert_0 > 10000:
    print("\n✅ SUCCESS: Certification correctly localizes attribution!")
    print("   The sparsification fix preserves per-cell localization through certification.")
else:
    print("\n❌ FAIL: Certification not localizing properly")
