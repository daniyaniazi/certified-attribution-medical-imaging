"""
PAPER IMPLEMENTATION GUIDE

This module maps paper equations to code implementations.
Reference: "Certified Pixel Attribution" paper

EQUATIONS & IMPLEMENTATIONS
============================

Eq. (1): Classifier
  f(x) → y_pred ∈ ℝ^C
  Implementation: src/models/factory.py - get_model()

Eq. (2): Attribution method
  h(f, x, y) → ℝ^{H×W}  (heatmap)
  Implementation: src/xai/attribution_unified.py
  - IntegratedGradientsUnified
  - GradCAMUnified
  - RISEUnified
  - OcclusionUnified

Eq. (3): Base heatmap
  heat(x) = h(f, x, y)  with y = argmax f(x)
  Implementation: attribute() method returns [H,W]

Eq. (4): Sparsification
  h_K(x) = TopK(heat(x), K%)
  Converts continuous heatmap to binary mask {0,1}
  Implementation: _sparsify_topk() in smoothing.py

Eq. (5): Certification via randomized smoothing
  cert[i] = argmax_{c∈{0,1}} P(h_K(x+ε) = c) where τ = threshold
  
  Algorithm (Steps 1-3 of certification):
  
  1. Sample n times (ε_t ~ N(0, σ²I)):
     For t=1..n:
       x_t = x + ε_t
       heat_t = h(f, x_t)
       mask_t = h_K(x_t) = TopK(heat_t, K%)
  
  2. Aggregate per-pixel (majority voting):
     p_1[i] = |{t : mask_t[i]=1}| / n
     p_0[i] = |{t : mask_t[i]=0}| / n = 1 - p_1[i]
  
  3. Threshold-based certification:
     If p_1[i] ≥ τ → cert[i] = 1 (certified important)
     Else if p_0[i] ≥ τ → cert[i] = 0 (certified unimportant)
     Else → cert[i] = ⊘ (abstain)
  
  Implementation: RandomizedSmoothingAttributor.certify() in smoothing.py

Eq. (7): Certified radius (Robustness guarantee)
  R = σ · Φ^(-1)(τ)
  
  Guarantees: For any perturbation δ with ||δ||_2 < R:
  The certified pixels (non-abstain) maintain same class
  
  Implementation: Computed in certify() using scipy.special.ndtri()

HYPERPARAMETER REFERENCE
========================

From paper (recommended defaults):
  σ (sigma):           0.15   # Gaussian noise std
  τ (tau):             0.75   # Certification threshold
  n (num_samples):     100    # Number of smoothing samples
  K (k_percent):       50, 30, 10  # Sparsification levels
  α (alpha):           0.001  # Significance level

WORKFLOW
========

Step 1: Train classifier
  python src/experiments/run_train.py --dataset chexpert --model resnet18

Step 2: Generate base attributions (Eq. 3)
  python src/experiments/run_attribution.py \
    --checkpoint outputs/checkpoints/chexpert/resnet18/best_model.pt \
    --method integrated_gradients

Step 3: Certify attributions (Eq. 4-7)
  python src/experiments/run_certify.py \
    --checkpoint outputs/checkpoints/chexpert/resnet18/best_model.pt \
    --sigma 0.15 \
    --tau 0.75 \
    --num-samples 100 \
    --k-percents 50,30,10

Step 4: Evaluate certified maps
  python src/experiments/run_eval.py \
    --checkpoint outputs/checkpoints/chexpert/resnet18/best_model.pt

OUTPUTS
=======

For each image & method combination:

1. Base heatmap (Eq. 3)
   {sample_id}_attr.npy → [H,W] ∈ [0,1]

2. Sparsified masks (Eq. 4) - for each K value
   {sample_id}_sparse_k50.npy → [H,W] ∈ {0,1}

3. Certified map (Eq. 5) - for each K value
   {sample_id}_certified_k50.npy → [H,W] ∈ {-1,0,1}
   where: 1 = certified important
          0 = certified unimportant
          -1 = abstain (⊘)

4. Probability maps (Eq. 5)
   {sample_id}_p1_k50.npy → [H,W] ∈ [0,1]
   {sample_id}_p0_k50.npy → [H,W] ∈ [0,1]

5. Metadata
   results.json containing:
   - certified_radius for each (σ,τ,K) combination
   - %certified, %abstained statistics
   - confidence metrics

PYTHON API EXAMPLE
==================

# Minimal example
from src.models.factory import get_model
from src.xai.attribution_unified import IntegratedGradientsUnified
from src.certify.smoothing import RandomizedSmoothingAttributor
import torch

# Load model
model, config = get_model('resnet18', num_classes=2)

# Load image
image = torch.randn(1, 3, 224, 224)

# Eq. (3): Get base heatmap
ig = IntegratedGradientsUnified(model)
heat = ig.attribute(image, target_class=1)  # → [H,W]

# Eq. (4): Sparsify
mask_k30 = (heat >= np.percentile(heat, 70)).astype(float)  # top 30%

# Eq. (5-7): Certify
smoother = RandomizedSmoothingAttributor(model, ig.attribute)
results = smoother.certify(
    image,
    k_percent=30,
    target_class=1,
    sigma=0.15,
    tau=0.75,
    num_samples=100
)

# Extract outputs
certified_map = results['certified_map']  # {-1,0,1}
p_1 = results['p_1']  # probability estimates
radius = results['certified_radius']  # R = σ·Φ^(-1)(τ)

print(f"Certified: {results['pct_certified']:.1f}%")
print(f"Radius: {radius:.4f}")

KEY INSIGHTS
============

1. Sparsification (Eq. 4):
   - Converts continuous attribution to binary segmentation
   - Same K-value used for all noisy samples
   - Top-K approach ensures consistent sparsity level

2. Randomized Smoothing (Eq. 5):
   - Samples noise at EACH iteration
   - Recomputes FULL attribution for each noisy input
   - Aggregates via simple majority voting per pixel
   - More computationally expensive but certified

3. Abstention (⊘):
   - Pixel certified iff confidence > τ for one class
   - Otherwise abstain (don't commit)
   - Allows model to reject low-confidence pixels

4. Certified Radius (Eq. 7):
   - R = σ · Φ^(-1)(τ) is from segmentation smoothing theorem
   - Larger τ → smaller R (stricter certification, smaller guarantee)
   - Larger σ → larger R (looser guarantee, fewer certified pixels)

COMPARISON WITH STANDARD ATTRIBUTION
====================================

Standard (non-certified):
  image → [attribute] → heatmap → visualization
  Fast, visual, but unstable to perturbations

Paper method (certified):
  image → [attribute] → heatmap
         → [sparsify] → binary mask
         → [smooth] → [vote] → certified map
  Slower, certifiably robust within radius R

Tradeoff: Robustness for computational cost
"""

# Example showing all steps
def full_certification_example():
    """Complete example matching paper framework."""
    import torch
    import numpy as np
    from src.models.factory import get_model
    from src.xai.attribution_unified import IntegratedGradientsUnified
    from src.certify.smoothing import RandomizedSmoothingAttributor
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 0: Load model & data
    model, _ = get_model('resnet18', num_classes=2, device=device)
    image = torch.randn(1, 3, 224, 224, device=device)
    
    # Step 1-2: Base attribution (Eq. 1-3)
    print("\n[Step 1-3] Computing base attribution...")
    ig = IntegratedGradientsUnified(model, device=device)
    heat = ig.attribute(image, target_class=1, num_steps=50)
    print(f"  Heatmap shape: {heat.shape}, range: [{heat.min():.3f}, {heat.max():.3f}]")
    
    # Step 3: Sparsification (Eq. 4)
    print("\n[Step 4] Sparsifying (K=30%)...")
    k_percent = 30
    threshold = np.percentile(heat.flatten(), 100 - k_percent)
    mask_k = (heat >= threshold).astype(np.float32)
    print(f"  Threshold: {threshold:.4f}, Sparsity: {100*mask_k.mean():.1f}%")
    
    # Step 4-6: Randomized smoothing (Eq. 5-7)
    print("\n[Step 5-7] Randomized smoothing certification...")
    smoother = RandomizedSmoothingAttributor(model, ig.attribute, device=device)
    results = smoother.certify(
        image,
        k_percent=k_percent,
        target_class=1,
        sigma=0.15,
        tau=0.75,
        num_samples=100
    )
    
    # Print results
    print("\n[Results]")
    print(f"  %Certified: {results['pct_certified']:.1f}%")
    print(f"  %Abstained: {results['pct_abstained']:.1f}%")
    print(f"  Certified Radius: {results['certified_radius']:.4f}")
    print(f"  (Guarantees robustness within L2 distance R)")
    
    return results


if __name__ == '__main__':
    print(__doc__)
    print("\nTo run full example:")
    print("  results = full_certification_example()")
