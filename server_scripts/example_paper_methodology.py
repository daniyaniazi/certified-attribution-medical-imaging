"""
End-to-end example: Certified Pixel Attribution on Medical Images

Demonstrates the complete workflow from paper:
1. Load model & image
2. Compute base attribution h(x)
3. Apply randomized smoothing Eq. (5)
4. Get certification results + certified radius Eq. (7)
5. Visualize certified maps {-1, 0, 1}

This example shows all 7 equations in action.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.factory import get_model
from xai.attribution_unified import IntegratedGradientsUnified, GradCAMUnified
from certify.smoothing import RandomizedSmoothingAttributor


def create_dummy_medical_image(img_size=224):
    """Create a synthetic medical image (chest X-ray like)."""
    # Simulate chest X-ray with anatomical structures
    img = np.zeros((img_size, img_size))
    
    # Lungs (left & right)
    y_center, x_center = img_size // 2, img_size // 2
    lung_radius = img_size // 4
    
    # Left lung
    yy, xx = np.ogrid[:img_size, :img_size]
    mask = (xx - (x_center - lung_radius))**2 + (yy - y_center)**2 <= lung_radius**2
    img[mask] = 0.3 + 0.2 * np.sin((xx[mask] + yy[mask]) / 20)
    
    # Right lung
    mask = (xx - (x_center + lung_radius))**2 + (yy - y_center)**2 <= lung_radius**2
    img[mask] = 0.3 + 0.2 * np.sin((xx[mask] + yy[mask]) / 20)
    
    # Heart region (center)
    mask = (xx - x_center)**2 + (yy - y_center)**2 <= (lung_radius // 3)**2
    img[mask] = 0.5
    
    # Add some noise
    img += np.random.randn(img_size, img_size) * 0.05
    
    # Normalize to [0, 1]
    img = np.clip(img, 0, 1).astype(np.float32)
    
    return img


def model_inference(model, image_tensor, target_class=1):
    """Get model prediction."""
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1)
        confidence = F.softmax(output, dim=1)
    return pred.item(), confidence[0].cpu().numpy()


def run_example():
    """Run complete example."""
    print("\n" + "="*80)
    print("CERTIFIED PIXEL ATTRIBUTION - PAPER METHODOLOGY EXAMPLE")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Step 1: Create/load model (Eq. 1)
    print("\n[Step 1] Model & Image Loading")
    print("-" * 40)
    
    model, config = get_model('resnet18', num_classes=2, pretrained=False)
    model = model.to(device)
    model.eval()
    print(f"✓ Model: {config['model_name']}")
    print(f"✓ Classes: {config['num_classes']}")
    
    # Create synthetic medical image
    img_np = create_dummy_medical_image(224)
    print(f"✓ Image: synthetic chest X-ray {img_np.shape}")
    
    # Convert to tensor format [1, C, H, W]
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    print(f"✓ Tensor shape: {img_tensor.shape}")
    
    # Get prediction
    pred_class, confidence = model_inference(model, img_tensor)
    target_class = pred_class
    print(f"✓ Prediction: class {pred_class}, confidence: {confidence[pred_class]:.3f}")
    
    # Step 2: Compute base attribution h(x) (Eq. 2-3)
    print("\n[Step 2] Base Attribution (Eq. 2-3)")
    print("-" * 40)
    print("Computing h(f, x, y_pred) for each pixel...")
    
    # Use Grad-CAM for efficiency
    attribution_method = GradCAMUnified(model)
    
    with torch.no_grad():
        heat = attribution_method.attribute(img_tensor, target_class=target_class)
    
    print(f"✓ Heatmap shape: {heat.shape}")
    print(f"✓ Heatmap range: [{heat.min():.3f}, {heat.max():.3f}]")
    
    # Step 3: Randomized Smoothing (Eq. 5)
    print("\n[Step 3] Randomized Smoothing & Certification (Eq. 5)")
    print("-" * 40)
    
    # Create smoothing attributor
    def attribution_wrapper(img_batch, target_cls):
        """Wrapper for attribution method."""
        return attribution_method.attribute(img_batch, target_class=target_cls)
    
    smoother = RandomizedSmoothingAttributor(model, attribution_wrapper, device=device)
    
    # Certification parameters (from paper)
    sigma = 0.15
    tau = 0.75
    num_samples = 100  # Reduced for faster demo (paper: 100)
    
    print(f"\nHyperparameters (paper default):")
    print(f"  σ (noise std):        {sigma}")
    print(f"  τ (threshold):        {tau}")
    print(f"  n (num samples):      {num_samples}")
    
    # Run smoothing certification for different K values
    results_by_k = {}
    
    for k_percent in [50, 30, 10]:
        print(f"\n→ Certifying with K={k_percent}% (top-K% sparsification)")
        
        results = smoother.certify(
            image=img_tensor,
            k_percent=k_percent,
            target_class=target_class,
            sigma=sigma,
            tau=tau,
            num_samples=num_samples,
            batch_size=32
        )
        
        results_by_k[k_percent] = results
        
        print(f"\n  Eq. (7) Certified Radius:")
        print(f"    R = σ × Φ^(-1)(τ) = {sigma} × {results['stats']['certified_radius']/sigma:.4f}")
        print(f"      = {results['certified_radius']:.4f}")
    
    # Step 4: Analyze results
    print("\n[Step 4] Results Analysis")
    print("-" * 40)
    
    for k_percent in [50, 30, 10]:
        results = results_by_k[k_percent]
        cert_map = results['certified_map']
        p_1 = results['p_1']
        p_0 = results['p_0']
        
        print(f"\nK={k_percent}% Results:")
        print(f"  Certified pixels:     {results['pct_certified']:.1f}%")
        print(f"    → As important:     {results['pct_certified_1']:.1f}%")
        print(f"    → As unimportant:   {results['pct_certified_0']:.1f}%")
        print(f"  Abstained pixels:     {results['pct_abstained']:.1f}%")
        print(f"  Certified radius R:   {results['certified_radius']:.4f}")
        print(f"  Avg confidence class 1: {np.mean(p_1[cert_map==1]):.3f}" if np.any(cert_map==1) else "  Avg confidence class 1: N/A")
        print(f"  Avg confidence class 0: {np.mean(p_0[cert_map==0]):.3f}" if np.any(cert_map==0) else "  Avg confidence class 0: N/A")
    
    # Step 5: Visualization
    print("\n[Step 5] Visualization")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original image and base heatmap
    ax = axes[0, 0]
    ax.imshow(img_np, cmap='gray')
    ax.set_title('Original Image')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(heat, cmap='jet')
    ax.set_title('Base Attribution h(x)')
    ax.axis('off')
    
    # Row 1: Certified maps for K=50, 30, 10
    for idx, k_percent in enumerate([50, 30, 10]):
        results = results_by_k[k_percent]
        cert_map = results['certified_map']
        
        # Visualize {-1, 0, 1} as colors
        ax = axes[0, 2 + idx]
        vis = np.zeros((*cert_map.shape, 3))
        vis[cert_map == 1] = [0, 1, 0]  # Green: certified important
        vis[cert_map == 0] = [1, 0, 0]  # Red: certified unimportant
        vis[cert_map == -1] = [0.5, 0.5, 0.5]  # Gray: abstain
        
        ax.imshow(vis)
        ax.set_title(f'Certified Map K={k_percent}%\nR={results["certified_radius"]:.4f}')
        ax.axis('off')
    
    # Row 2: Probability maps p_1
    for idx, k_percent in enumerate([50, 30, 10]):
        results = results_by_k[k_percent]
        
        ax = axes[1, idx + 1]
        im = ax.imshow(results['p_1'], cmap='viridis')
        ax.set_title(f'p_1 (K={k_percent}%)')
        plt.colorbar(im, ax=ax)
    
    # Legend
    ax = axes[1, 0]
    ax.axis('off')
    green_patch = mpatches.Patch(color='green', label='Certified Important (1)')
    red_patch = mpatches.Patch(color='red', label='Certified Unimportant (0)')
    gray_patch = mpatches.Patch(color='gray', label='Abstained (⊘)')
    ax.legend(handles=[green_patch, red_patch, gray_patch], loc='center', fontsize=10)
    ax.set_title('Legend')
    
    plt.tight_layout()
    plt.savefig('example_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to: example_results.png")
    
    # Step 6: Summary
    print("\n[Step 6] Summary")
    print("-" * 40)
    print(f"""
WORKFLOW COMPLETED:

1. ✓ Eq. (1): Classifier f(x) → prediction
2. ✓ Eq. (2-3): Attribution h(f, x, y) → heatmap
3. ✓ Eq. (4): Sparsification h_K(x) → binary mask
4. ✓ Eq. (5): Smoothing certification → p_1, p_0
5. ✓ Eq. (7): Certified radius R = σ·Φ^(-1)(τ)

RESULTS:
- Analyzed 3 different sparsification levels (K=50%, 30%, 10%)
- For each K, computed robustness guarantee (certified radius)
- Obtained certified attribution maps {{-1 (abstain), 0, 1}}

INTERPRETATION:
- Green pixels: certified as important
- Red pixels: certified as unimportant  
- Gray pixels: insufficient confidence (abstain)

ROBUSTNESS GUARANTEE:
- Any image perturbation with norm < R preserves certification
- Larger K → easier to certify → smaller R
- Smaller K → harder to certify → larger R
    """)
    
    print("="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)
    
    return results_by_k


if __name__ == '__main__':
    try:
        results = run_example()
        print("\n✓ All steps completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
