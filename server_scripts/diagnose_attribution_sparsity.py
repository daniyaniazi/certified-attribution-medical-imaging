#!/usr/bin/env python
"""Diagnose why IG/GradCAM/RISE produce sparse certifications.

Visualize:
1. Raw attribution heatmaps (before sparsification)
2. Histogram of attribution values
3. Effect of noise on top-K pixel rankings
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.grid_dataset import GridDataset
from src.models.grid_multihead import GridMultiHead
from src.xai.attribution_unified import (
    IntegratedGradientsUnified,
    GradCAMUnified,
    RISEUnified,
    OcclusionUnified,
    LRPUnified,
)


def get_resnet_target_layer(model):
    """Get target layer for GradCAM."""
    try:
        return model.backbone.layer4[-1]
    except Exception:
        raise ValueError("Could not find target layer for GradCAM")


class DiFull_Wrapper(torch.nn.Module):
    """DiFull wrapper for full-grid attribution."""
    def __init__(self, grid_model, head_id):
        super().__init__()
        self.grid_model = grid_model
        self.head_id = head_id
    
    def forward(self, x):
        logits = self.grid_model(x)
        return logits[self.head_id].unsqueeze(0)


def normalize_for_viz(heatmap):
    """Normalize for visualization only."""
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_min >= 0 and h_max <= 1:
        return heatmap
    if h_max - h_min < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - h_min) / (h_max - h_min)


def analyze_attribution_sparsity(grid_pt_path: str, device: str = "cpu"):
    """Analyze raw attribution distributions and noise sensitivity."""
    
    # Load dataset and model
    grid_ds = GridDataset(Path(grid_pt_path))
    num_heads = grid_ds.scale * grid_ds.scale
    
    model = GridMultiHead(
        "resnet18",
        num_classes=8,
        num_heads=num_heads,
        pretrained=True,
        scale=grid_ds.scale,
    )
    model.to(device)
    model.eval()
    
    # Get first sample
    sample = grid_ds[0]
    image = sample["image"].unsqueeze(0).to(device)
    head_id = 0  # Bottom-left cell (Cell 0)
    target_class = sample["labels"][head_id].item()
    
    # Build attribution methods
    wrapper = DiFull_Wrapper(model, head_id)
    target_layer = get_resnet_target_layer(model)
    
    methods = {
        "IntegratedGradients": IntegratedGradientsUnified(wrapper, device),
        "GradCAM": GradCAMUnified(wrapper, target_layer, device),
        "RISE": RISEUnified(wrapper, device),
        "Occlusion": OcclusionUnified(wrapper, device),
        "LRP": LRPUnified(wrapper, device),
    }
    
    # Compute clean attributions
    print("Computing clean attributions...")
    clean_attrs = {}
    for name, method in methods.items():
        print(f"  {name}...")
        attr = method.attribute(image, target_class)
        clean_attrs[name] = attr
    
    # Analyze distributions
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 5, hspace=0.3, wspace=0.3)
    
    method_names = list(methods.keys())
    
    for idx, name in enumerate(method_names):
        attr = clean_attrs[name]
        attr_viz = normalize_for_viz(attr)
        
        # Row 0: Heatmap visualization
        ax_heat = fig.add_subplot(gs[0, idx])
        im = ax_heat.imshow(attr_viz, cmap='inferno', vmin=0, vmax=1)
        ax_heat.set_title(f"{name}\nRaw Attribution", fontsize=10, fontweight='bold')
        ax_heat.axis('off')
        plt.colorbar(im, ax=ax_heat, fraction=0.046)
        
        # Row 1: Histogram of values
        ax_hist = fig.add_subplot(gs[1, idx])
        attr_flat = attr.flatten()
        nonzero = attr_flat[attr_flat > 1e-8]
        
        ax_hist.hist(attr_flat, bins=50, alpha=0.7, color='blue', label='All pixels')
        if len(nonzero) > 0:
            ax_hist.hist(nonzero, bins=50, alpha=0.7, color='red', label='Non-zero')
        ax_hist.set_xlabel('Attribution value')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title(f'Distribution\n{len(nonzero)}/{len(attr_flat)} non-zero')
        ax_hist.legend(fontsize=8)
        ax_hist.set_yscale('log')
        
        # Row 2: Statistics
        ax_stats = fig.add_subplot(gs[2, idx])
        ax_stats.axis('off')
        
        stats_text = (
            f"Min: {attr.min():.6f}\n"
            f"Max: {attr.max():.6f}\n"
            f"Mean: {attr.mean():.6f}\n"
            f"Std: {attr.std():.6f}\n"
            f"Median: {np.median(attr):.6f}\n"
            f"\n"
            f"Non-zero: {len(nonzero)}/{len(attr_flat)}\n"
            f"Non-zero %: {100*len(nonzero)/len(attr_flat):.1f}%\n"
            f"\n"
            f"Top 1%: {np.percentile(attr, 99):.6f}\n"
            f"Top 5%: {np.percentile(attr, 95):.6f}\n"
            f"Top 10%: {np.percentile(attr, 90):.6f}\n"
        )
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Raw Attribution Analysis (Before Certification)', fontsize=14, fontweight='bold')
    out_path = Path("outputs/diagnostics/attribution_sparsity_analysis.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis to: {out_path}")
    plt.close()
    
    # Now analyze noise sensitivity
    print("\nAnalyzing noise sensitivity...")
    analyze_noise_sensitivity(image, wrapper, target_class, methods, device)


def analyze_noise_sensitivity(image, model_wrapper, target_class, methods, device, sigma=0.15, n_samples=10):
    """Check how noise affects top-K pixel rankings."""
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    method_names = list(methods.keys())
    
    for idx, name in enumerate(method_names):
        method = methods[name]
        
        # Get clean attribution
        clean_attr = method.attribute(image, target_class)
        h, w = clean_attr.shape
        
        # Track which pixels stay in top-30% across noisy samples
        k_percent = 30
        k_count = int(np.ceil(h * w * k_percent / 100.0))
        
        vote_in_topk = np.zeros((h, w), dtype=np.int32)
        
        for i in range(n_samples):
            # Add noise
            noise = torch.randn_like(image) * sigma
            noisy_image = torch.clamp(image + noise, 0, 1)
            
            # Get noisy attribution
            noisy_attr = method.attribute(noisy_image, target_class)
            
            # Check top-K
            flat = noisy_attr.flatten()
            if k_count <= len(flat):
                threshold = np.partition(flat, -k_count)[-k_count]
                topk_mask = noisy_attr >= threshold
                vote_in_topk += topk_mask.astype(np.int32)
        
        # Compute stability: how often each pixel stayed in top-K
        stability = vote_in_topk / n_samples
        
        # Row 0: Clean attribution
        ax = axes[0, idx]
        clean_viz = normalize_for_viz(clean_attr)
        ax.imshow(clean_viz, cmap='inferno', vmin=0, vmax=1)
        ax.set_title(f"{name}\nClean", fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Row 1: Stability under noise
        ax = axes[1, idx]
        im = ax.imshow(stability, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Top-{k_percent}% Stability\n({n_samples} noisy samples)", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='P(in top-K)')
    
    plt.suptitle(f'Noise Sensitivity Analysis (σ={sigma}, n={n_samples})', fontsize=14, fontweight='bold')
    out_path = Path("outputs/diagnostics/noise_sensitivity_analysis.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved noise sensitivity analysis to: {out_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--grid_pt", default="data/raw/grid/isic/grid.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    
    analyze_attribution_sparsity(args.grid_pt, args.device)
