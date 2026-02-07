#!/usr/bin/env python
"""
Quick sanity check: Verify the fixed sparsification works in RandomizedSmoothingAttributor
"""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.certify.smoothing import RandomizedSmoothingAttributor

print("="*70)
print("SPARSIFICATION FIX: SANITY CHECK")
print("="*70)

# Create a simple attribution function that returns sparse heatmaps
def mock_attribution_sparse(image_batch, target_class):
    """Mock attribution that produces DiFull-like sparse heatmaps."""
    # Return heatmap: Cell 0 concentrated, Cells 1-3 zeros
    h, w = image_batch.shape[-2:]
    hmap = np.zeros((h, w))
    hmap[0:h//2, 0:w//2] = np.random.uniform(0.05, 0.08, (h//2, w//2))
    hmap[0:h//2, w//2:] = np.random.uniform(0, 1e-5, (h//2, w//2))
    hmap[h//2:, 0:w//2] = np.random.uniform(0, 1e-5, (h//2, w//2))
    hmap[h//2:, w//2:] = np.random.uniform(0, 1e-5, (h//2, w//2))
    return hmap

# Create dummy image and smoother
device = "cpu"
image = torch.randn(1, 3, 256, 256).clamp(0, 1)
smoother = RandomizedSmoothingAttributor(None, None, device=device)
smoother.attribution_func = mock_attribution_sparse

# Run certification with the fixed sparsification
print("\nRunning certification with fixed sparsification...")
print("  - Image size: 256x256")
print("  - Num cells: 4 (2x2 grid)")
print("  - Attribution: Cell 0 concentrated (0.05-0.08), Cells 1-3 zeros")
print("  - Num samples: 50")
print("  - K%: 50")

try:
    results = smoother.certify(
        image,
        k_percent=50,
        target_class=0,
        sigma=0.15,
        num_samples=50,
        tau=0.75,
        alpha=0.001,
        batch_size=10,
        save_noisy_samples=False
    )
    
    cert_map = results['certified_map']
    
    # Analyze results
    c0 = cert_map[0:128, 0:128]
    c1 = cert_map[0:128, 128:]
    
    c0_cert_1 = (c0 == 1).sum()
    c0_abstain = (c0 == -1).sum()
    c1_cert_0 = (c1 == 0).sum()
    c1_abstain = (c1 == -1).sum()
    
    print("\n✅ CERTIFICATION COMPLETED SUCCESSFULLY")
    print("\nResults:")
    print(f"  Cell 0 (target):")
    print(f"    - Certified as '1': {c0_cert_1:5d} / 16384 ({100*c0_cert_1/16384:5.1f}%)")
    print(f"    - Abstained:        {c0_abstain:5d} / 16384 ({100*c0_abstain/16384:5.1f}%)")
    print(f"  Cell 1 (non-target):")
    print(f"    - Certified as '0': {c1_cert_0:5d} / 16384 ({100*c1_cert_0/16384:5.1f}%)")
    print(f"    - Abstained:        {c1_abstain:5d} / 16384 ({100*c1_abstain/16384:5.1f}%)")
    
    # Validation
    print("\n" + "="*70)
    if c0_cert_1 > 5000:
        print("✅ Cell 0 sufficiently certified as '1'")
    else:
        print(f"⚠️  Cell 0 only {c0_cert_1} pixels certified as '1' (expected > 5000)")
    
    if c1_cert_0 > 1000 or c1_abstain > 10000:
        print("✅ Cell 1 properly differentiated (certified '0' or abstained)")
    else:
        print("⚠️  Cell 1 may not be properly differentiated")
    
    if c0_cert_1 > 5000:
        print("\n✅ SPARSIFICATION FIX VERIFIED: Works correctly in certification pipeline!")
    else:
        print("\n⚠️  May need more samples or adjustment to parameters")
    
except Exception as e:
    print(f"\n❌ ERROR during certification: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
