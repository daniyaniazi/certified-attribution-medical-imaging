"""Visualization utilities for attribution maps."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import RdBu_r, viridis
from pathlib import Path
from typing import Tuple, Optional


def save_attribution_heatmap(
    image: np.ndarray,
    attribution: np.ndarray,
    output_path: str,
    title: str = "",
    alpha: float = 0.6,
    cmap: str = 'RdBu_r'
):
    """
    Save attribution heatmap overlay on image.
    
    Args:
        image: input image [H,W,3] or [H,W] in [0,1] or [0,255]
        attribution: attribution map [H,W] in [0,1]
        output_path: where to save the figure
        title: figure title
        alpha: transparency of overlay
        cmap: colormap name
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize image to [0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Only normalize if values are outside [0,1] (i.e., LRP)
    # Other methods (IG, GradCAM, RISE, Occlusion) already return [0,1] values
    attr_min, attr_max = attribution.min(), attribution.max()
    if attr_min >= 0 and attr_max <= 1:
        # Already normalized, keep as-is
        attribution_viz = attribution
    elif attr_max - attr_min > 1e-8:
        # Unnormalized (LRP), normalize to [0,1]
        attribution_viz = (attribution - attr_min) / (attr_max - attr_min)
    else:
        attribution_viz = np.zeros_like(attribution)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray' if image.shape[-1] == 1 else None)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Attribution map
    im = axes[1].imshow(attribution_viz, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Attribution Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(attribution_viz, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def save_certified_map(
    certified: np.ndarray,
    output_path: str,
    title: str = ""
):
    """
    Save certified attribution map.
    
    Args:
        certified: certified map with {-1 (abstain), 0, 1}
        output_path: where to save
        title: figure title
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create custom colormap: -1=gray, 0=red, 1=green
    cmap_colors = ['gray', 'red', 'green']
    cmap_data = np.array([
        [0.5, 0.5, 0.5],    # -1: gray
        [1.0, 0.0, 0.0],    # 0: red
        [0.0, 1.0, 0.0]     # 1: green
    ])
    
    # Shift certified values for colormap
    certified_plot = certified + 1  # -1->0, 0->1, 1->2
    
    im = ax.imshow(certified_plot, cmap='RdYlGn', vmin=0, vmax=2)
    ax.set_title(f"Certified Attribution{' - ' + title if title else ''}")
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Abstain', 'Class 0', 'Class 1'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_deletion_curve(
    deletions: list,
    output_path: str,
    title: str = "",
    label: str = "Faithfulness"
):
    """
    Plot deletion curve (faithfulness metric).
    
    Args:
        deletions: list of confidence values as pixels are deleted
        output_path: where to save
        title: figure title
        label: curve label
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    percentages = np.linspace(0, 100, len(deletions))
    ax.plot(percentages, deletions, marker='o', label=label, linewidth=2)
    ax.fill_between(percentages, deletions, alpha=0.3)
    
    ax.set_xlabel("Percentage of Pixels Deleted")
    ax.set_ylabel("Model Confidence")
    ax.set_title(f"Deletion Curve{' - ' + title if title else ''}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
