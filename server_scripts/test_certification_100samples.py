#!/usr/bin/env python
"""Certification test with proper sample count."""

import numpy as np

print("=== CERTIFICATION WITH REALISTIC SAMPLE COUNT ===\n")

# Simulate realistic DiFull attribution across N noisy samples
num_samples = 100
np.random.seed(42)

masks_by_sample = []

# Each sample: Cell 0 has ~95% top-K, Cells 1-3 have ~35% top-K due to sparsification
# (as shown in earlier ranking test)
for sample_id in range(num_samples):
    mask = np.zeros((256, 256))
    
    # Cell 0: 95-98% marked as "1" (top-K includes most of Cell 0)
    mask[0:128, 0:128] = np.random.binomial(1, 0.96, (128, 128))
    
    # Cells 1-3: 30-35% marked as "1" (as secondary pixels in top-K ranking)
    mask[0:128, 128:] = np.random.binomial(1, 0.33, (128, 128))
    mask[128:, 0:128] = np.random.binomial(1, 0.33, (128, 128))
    mask[128:, 128:] = np.random.binomial(1, 0.33, (128, 128))
    
    masks_by_sample.append(mask)

# Aggregate via majority voting
masks_array = np.array(masks_by_sample)
p_1 = np.mean(masks_array, axis=0)
p_0 = 1.0 - p_1

print(f"After {num_samples} samples:")
print(f"  p_1[Cell 0]: mean={p_1[0:128, 0:128].mean():.4f}, min={p_1[0:128, 0:128].min():.4f}, max={p_1[0:128, 0:128].max():.4f}")
print(f"  p_1[Cell 1]: mean={p_1[0:128, 128:].mean():.4f}")
print(f"  p_0[Cell 0]: mean={p_0[0:128, 0:128].mean():.4f}")
print(f"  p_0[Cell 1]: mean={p_0[0:128, 128:].mean():.4f}")

# Simple certification (without Clopper-Pearson for clarity)
tau = 0.75
certified = np.full((256, 256), -1, dtype=np.int8)
certified[p_1 >= tau] = 1
certified[p_0 >= tau] = 0

print(f"\nCertification with τ={tau}:")

for cell_id in range(4):
    r = (cell_id // 2) * 128
    c = (cell_id % 2) * 128
    cert_region = certified[r:r+128, c:c+128]
    
    cert_1 = (cert_region == 1).sum()
    cert_0 = (cert_region == 0).sum()
    abstain = (cert_region == -1).sum()
    
    print(f"  Cell {cell_id}: Cert-1={cert_1:5d} ({100*cert_1/16384:5.1f}%), "
          f"Cert-0={cert_0:5d} ({100*cert_0/16384:5.1f}%), "
          f"Abstain={abstain:5d} ({100*abstain/16384:5.1f}%)")

# Validation
c0_cert_1 = (certified[0:128, 0:128] == 1).sum()
c1_cert_0 = (certified[0:128, 128:] == 0).sum()

print("\n=== RESULT ===")
if c0_cert_1 > 15000:
    print(f"✓ Cell 0 heavily certified as '1' ({c0_cert_1}/16384)")
else:
    print(f"✗ Cell 0 not certified as '1' ({c0_cert_1}/16384)")

if c1_cert_0 > 15000:
    print(f"✓ Cell 1 heavily certified as '0' ({c1_cert_0}/16384)")
else:
    print(f"✗ Cell 1 not fully certified as '0' ({c1_cert_0}/16384), "
          f"due to {(certified[0:128, 128:] == -1).sum()} abstentions")

if c0_cert_1 > 15000 and c1_cert_0 > 10000:
    print("\n✅ With 100 samples, certification properly localizes!")
else:
    print("\n⚠️  More samples may be needed for full localization")
