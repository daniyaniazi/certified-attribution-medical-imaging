#!/usr/bin/env python
"""Grid ISIC Certification Server - multi-head per-cell attribution certification."""

import sys
from pathlib import Path
import argparse
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.grid_dataset import GridDataset
from src.models.grid_multihead import GridMultiHead
from src.certify.smoothing import RandomizedSmoothingAttributor
from src.xai.attribution_unified import (
    IntegratedGradientsUnified,
    GradCAMUnified,
    RISEUnified,
    OcclusionUnified,
    LRPUnified,
)


def normalize_for_viz(heatmap):
    """Only normalize if values are outside [0,1] range (i.e., LRP)."""
    if heatmap is None:
        return None
    h_min, h_max = heatmap.min(), heatmap.max()
    # If already in [0,1], keep as-is (IG, GradCAM, RISE, Occlusion)
    if h_min >= 0 and h_max <= 1:
        return heatmap
    # Otherwise normalize (LRP)
    if h_max - h_min < 1e-8:
        return np.zeros_like(heatmap)
    return (heatmap - h_min) / (h_max - h_min)


def _tensor_to_hwc(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().clamp(0, 1).numpy()
    if img.ndim == 4:
        img = img[0]
    return np.transpose(img, (1, 2, 0))


def save_artifacts(base_dir: Path, model_name: str, method_name: str, k: int, img_idx: int,
                   image_tensor: torch.Tensor, results: dict):
    """Persist per-image arrays (.npy) and quick PNG overlays."""
    arr_dir = base_dir / "arrays" / model_name / method_name / f"k_{k}"
    viz_dir = base_dir / "viz" / model_name / method_name / f"k_{k}"
    arr_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    clean = results.get("heatmap_clean")
    ss_map = results.get("ss_map")
    c_map = results.get("certified_map")
    noisy_samples = results.get("sample_noisy_heatmaps", [])

    # Save arrays
    if clean is not None:
        np.save(arr_dir / f"img_{img_idx}_clean.npy", clean)
    if ss_map is not None:
        np.save(arr_dir / f"img_{img_idx}_ssmap.npy", ss_map)
    if c_map is not None:
        np.save(arr_dir / f"img_{img_idx}_certmap.npy", c_map)
    if noisy_samples:
        np.save(arr_dir / f"img_{img_idx}_noisy_sample0.npy", noisy_samples[0])

    # Build base image from tensor
    img = _tensor_to_hwc(image_tensor)

    # Visualize overlays
    def _save_overlay(data, cmap, title, fname, vmin=None, vmax=None):
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        if data is not None:
            plt.imshow(data, cmap=cmap, alpha=0.5, vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(viz_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()

    # Normalize clean heatmap only if needed (LRP is unnormalized)
    clean_viz = normalize_for_viz(clean)
    _save_overlay(clean_viz, "inferno", f"Clean heatmap (K={k}%)", f"img_{img_idx}_clean.png")
    _save_overlay(ss_map, "magma", "Smoothed sparsified map", f"img_{img_idx}_ssmap.png", vmin=0, vmax=1)

    if c_map is not None:
        viz_rgb = np.ones((c_map.shape[0], c_map.shape[1], 3))
        viz_rgb[c_map == 0] = [1.0, 1.0, 1.0]
        viz_rgb[c_map == 1] = [1.0, 0.55, 0.0]
        viz_rgb[c_map == -1] = [0.8, 0.8, 0.8]
        _save_overlay(viz_rgb, None, "Certified map (1=orange,0=white,abstain=gray)", f"img_{img_idx}_certmap.png")


def generate_figure4_style_panel(save_dir: Path, model_name: str, all_images_results: list):
    """Create Figure 4 style panel: rows=images (up to 5), cols=methods with SS and Certified pairs."""
    save_dir.mkdir(parents=True, exist_ok=True)
    methods_order = ["IntegratedGradients", "GradCAM", "RISE", "Occlusion", "LRP"]
    
    # Limit to 5 images
    all_images_results = all_images_results[:5]
    n_rows = len(all_images_results)
    n_cols = len(methods_order) * 2  # SS and Certified for each method
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, img_data in enumerate(all_images_results):
        img = _tensor_to_hwc(img_data['image_tensor'])
        method_results_map = img_data['method_results_map']
        
        for method_idx, mname in enumerate(methods_order):
            res_by_k = method_results_map.get(mname, {})
            col_ss = method_idx * 2
            col_cert = method_idx * 2 + 1
            
            # SS column
            ax_ss = axes[row_idx, col_ss]
            ss_map = None
            if res_by_k:
                ss_map = next(iter(res_by_k.values())).get("ss_map")
            ax_ss.imshow(ss_map if ss_map is not None else np.zeros(img.shape[:2]), cmap="gray", vmin=0, vmax=1)
            if row_idx == 0:
                ax_ss.set_title(f'{mname}\nSS', fontsize=10, fontweight='bold')
            ax_ss.axis("off")
            
            # Certified column (overlayed K values)
            ax_cert = axes[row_idx, col_cert]
            c50 = res_by_k.get(50, {}).get("certified_map")
            c25 = res_by_k.get(25, {}).get("certified_map")
            c5 = res_by_k.get(5, {}).get("certified_map")
            
            if c50 is None:
                c50 = np.zeros(img.shape[:2])
            if c25 is None:
                c25 = np.zeros(img.shape[:2])
            if c5 is None:
                c5 = np.zeros(img.shape[:2])
            
            # Create overlay map with priority: K=5% > K=25% > K=50%
            overlay_map = np.ones(img.shape[:2] + (3,)) * 0.9  # light gray background
            overlay_map[c50 == 1] = [1.0, 0.65, 0.0]  # orange for top 50%
            overlay_map[c25 == 1] = [0.9, 0.3, 0.3]  # red for top 25%
            overlay_map[c5 == 1] = [0.2, 0.0, 0.2]  # dark purple for top 5%
            
            ax_cert.imshow(overlay_map)
            if row_idx == 0:
                ax_cert.set_title(f'Certified', fontsize=10, fontweight='bold')
            ax_cert.axis("off")
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=[1.0, 0.65, 0.0], label='Top 50%'),
        mpatches.Patch(color=[0.9, 0.3, 0.3], label='Top 25%'),
        mpatches.Patch(color=[0.2, 0.0, 0.2], label='Top 5%'),
        mpatches.Patch(color=[0.9, 0.9, 0.9], label='Not certified'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.01), fontsize=11)
    
    plt.suptitle(f'{model_name}: Overlayed Certified Attributions (Figure 4 Style)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    out_path = save_dir / f"{model_name}_figure4_style.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def generate_paper_panel_all_methods(save_dir: Path, model_name: str, img_idx: int,
                                     img_tensor: torch.Tensor, method_results_map: dict):
    """Create paper-style panel matching Figure 2: rows=methods, cols=Input|SS|K=50/25/5|Overlayed."""
    save_dir.mkdir(parents=True, exist_ok=True)
    methods_order = ["IntegratedGradients", "GradCAM", "RISE", "Occlusion", "LRP"]
    img = _tensor_to_hwc(img_tensor)
    n_rows = len(methods_order)
    n_cols = 6  # Input, SS, K=50%, K=25%, K=5%, Overlayed
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.5 * n_rows))
    axes = np.atleast_2d(axes)

    for r, mname in enumerate(methods_order):
        res_by_k = method_results_map.get(mname, {})
        
        # Column 0: Input image
        ax_input = axes[r, 0]
        ax_input.imshow(img)
        if r == 0:
            ax_input.set_title('Input', fontsize=12, fontweight='bold')
        # Method name label on the left
        ax_input.text(-0.1, 0.5, mname, transform=ax_input.transAxes,
                     fontsize=11, fontweight='bold', va='center', ha='right', rotation=0)
        ax_input.axis("off")
        
        # Column 1: SS (Smoothed Sparsified)
        ax_ss = axes[r, 1]
        ss_map = None
        # Pick any available K to fetch ss_map
        if res_by_k:
            ss_map = next(iter(res_by_k.values())).get("ss_map")
        ax_ss.imshow(ss_map if ss_map is not None else np.zeros(img.shape[:2]), cmap="gray", vmin=0, vmax=1)
        if r == 0:
            ax_ss.set_title('SS', fontsize=12, fontweight='bold')
        ax_ss.axis("off")

        # Columns 2-4: K=50%, 25%, 5% certified maps
        for ci, k in enumerate([50, 25, 5], start=2):
            axk = axes[r, ci]
            res = res_by_k.get(k, {})
            c_map = res.get("certified_map")
            if c_map is not None:
                # Color-code: certified-1=orange, certified-0=white, abstain=gray
                viz_map = np.ones((c_map.shape[0], c_map.shape[1], 3))
                viz_map[c_map == 0] = [1.0, 1.0, 1.0]  # white
                viz_map[c_map == 1] = [1.0, 0.65, 0.0]  # orange
                viz_map[c_map == -1] = [0.85, 0.85, 0.85]  # light gray
                axk.imshow(viz_map)
            else:
                axk.imshow(np.ones(img.shape[:2] + (3,)) * 0.85)
            if r == 0:
                axk.set_title(f'K={k}%', fontsize=12, fontweight='bold')
            axk.axis("off")

        # Column 5: Overlayed (combine all K certified pixels)
        axo = axes[r, 5]
        c50 = res_by_k.get(50, {}).get("certified_map")
        c25 = res_by_k.get(25, {}).get("certified_map")
        c5 = res_by_k.get(5, {}).get("certified_map")
        if c50 is None:
            c50 = np.zeros(img.shape[:2])
        if c25 is None:
            c25 = np.zeros(img.shape[:2])
        if c5 is None:
            c5 = np.zeros(img.shape[:2])
        
        # Create overlay map with priority: K=5% > K=25% > K=50%
        overlay_map = np.ones(img.shape[:2] + (3,)) * 0.9  # light gray background
        overlay_map[c50 == 1] = [1.0, 0.65, 0.0]  # orange for top 50%
        overlay_map[c25 == 1] = [0.9, 0.3, 0.3]  # red for top 25%
        overlay_map[c5 == 1] = [0.2, 0.0, 0.2]  # dark purple for top 5%
        
        axo.imshow(overlay_map)
        if r == 0:
            axo.set_title('Overlayed', fontsize=12, fontweight='bold')
        axo.axis("off")

    # Add "Certified" header above K columns
    fig.text(0.63, 0.97, 'Certified', ha='center', fontsize=13, fontweight='bold')
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=[1.0, 0.65, 0.0], label='Top 50%'),
        mpatches.Patch(color=[0.9, 0.3, 0.3], label='Top 25%'),
        mpatches.Patch(color=[0.2, 0.0, 0.2], label='Top 5%'),
        mpatches.Patch(color=[0.9, 0.9, 0.9], label='Not certified'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.suptitle(f'{model_name}: Certified Attributions (Paper Style) - Image {img_idx}', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    out_path = save_dir / f"{model_name}_img_{img_idx}_paper_panel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def get_resnet_target_layer(model: GridMultiHead):
    # For ResNet backbones: last conv in layer4
    try:
        layer4 = model.feature_extractor.body[6]  # layer4 module
        return layer4[-1].conv2 if hasattr(layer4[-1], "conv2") else layer4[-1]
    except Exception:
        # Fallback: last Conv2d
        for _, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
    raise ValueError("Could not find target layer for GradCAM")


class DiFull_Wrapper(nn.Module):
    """DiFull-style wrapper: processes only the target cell through the backbone.
    
    This implements the paper's DiFull approach where each cell is passed separately
    through the backbone, ensuring true disconnection between cells. The wrapper
    extracts the target cell from the full grid, processes it independently, and
    applies only the corresponding head.
    """
    def __init__(self, grid_model: GridMultiHead, head_id: int, target_cell: int, scale: int):
        super().__init__()
        self.grid_model = grid_model
        self.head_id = head_id
        self.target_cell = target_cell
        self.scale = scale

    def forward(self, x: torch.Tensor):
        """Forward pass with DiFull disconnection.
        
        Args:
            x: [B, C, H*scale, W*scale] - full grid image
        Returns:
            [B, num_classes] - logits for target cell only
        """
        B, C, full_H, full_W = x.shape
        cell_H = full_H // self.scale
        cell_W = full_W // self.scale
        
        # Extract only the target cell (DiFull disconnection)
        row = self.target_cell // self.scale
        col = self.target_cell % self.scale
        cell = x[:, :, row*cell_H:(row+1)*cell_H, col*cell_W:(col+1)*cell_W]
        
        # Pass only this cell through the backbone
        features = self.grid_model.feature_extractor(cell)
        
        # Apply the corresponding head
        logits = self.grid_model.heads[self.head_id](features)
        
        return logits


def build_attr_methods(grid_model: GridMultiHead, device: str, head_id: int, target_cell: int, scale: int):
    """Build attribution methods with DiFull wrapper for true cell disconnection."""
    wrapper = DiFull_Wrapper(grid_model, head_id, target_cell, scale)
    target_layer = get_resnet_target_layer(grid_model)
    return {
        "GradCAM": GradCAMUnified(wrapper, target_layer, device),
        "IntegratedGradients": IntegratedGradientsUnified(wrapper, device),
        "RISE": RISEUnified(wrapper, device),
        "Occlusion": OcclusionUnified(wrapper, device),
        "LRP": LRPUnified(wrapper, device, epsilon=1e-6),
    }


def parse_args():
    p = argparse.ArgumentParser("Grid ISIC Certification")
    p.add_argument("--grid_pt", required=True, help="Path to grid.pt generated by generate_grid_dataset")
    p.add_argument("--checkpoint", required=False, default=None, help="Optional single-head checkpoint to init backbone weights")
    p.add_argument("--num_classes", type=int, default=8)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--tau", type=float, default=0.75)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--k_percents", type=int, nargs="+", default=[50, 25, 5])
    p.add_argument("--save_dir", default=str(ROOT / "outputs" / "certifications" / "grid_isic"))
    p.add_argument("--heatmap_dir", default=str(ROOT / "outputs" / "bulk_certifcation" / "grid" / "isic" / "resnet18" / "certified_maps"))
    p.add_argument("--panel_examples", type=int, default=3, help="Number of images to generate paper-style panels for")
    p.add_argument("--save_noisy_samples", action="store_true")
    p.add_argument("--max_noisy_samples", type=int, default=3)
    return p.parse_args()


def load_model(num_heads: int, num_classes: int, device: str, checkpoint: str = None) -> GridMultiHead:
    model = GridMultiHead("resnet18", num_classes=num_classes, num_heads=num_heads, pretrained=(checkpoint is None))
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model_dict = model.state_dict()
        state = {k.replace("module.", ""): v for k, v in state.items() if k.replace("module.", "") in model_dict}
        model_dict.update(state)
        model.load_state_dict(model_dict, strict=False)
        model.duplicate_head_weights(0)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = args.device

    # Load grid dataset
    grid_ds = GridDataset(Path(args.grid_pt))
    num_heads = grid_ds.scale * grid_ds.scale
    loader = DataLoader(grid_ds, batch_size=1, shuffle=False)

    # Model
    model = load_model(num_heads=num_heads, num_classes=args.num_classes, device=device, checkpoint=args.checkpoint)

    save_dir = Path(args.save_dir)
    heatmap_dir = Path(args.heatmap_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    all_images_for_fig4 = []
    panel_count = 0
    
    for batch_idx, sample in enumerate(tqdm(loader, desc="Certifying grids (DiFull)")):
        image = sample["image"].to(device)
        target_class = int(sample["target_class"].item())
        head_id = int(sample["target_head"].item())
        target_cell = int(sample["target_head"].item())  # same as head_id in our case
        scale = int(sample["meta"]["scale"].item()) if isinstance(sample["meta"], dict) and "scale" in sample["meta"] else grid_ds.scale

        # DiFull: cells are truly disconnected via separate forward passes in the wrapper
        # No manual masking needed - the wrapper extracts and processes only the target cell
        methods = build_attr_methods(model, device, head_id, target_cell, scale)
        smoother = RandomizedSmoothingAttributor(model, None, device=device)

        # Build attribution wrappers
        def make_attr_wrapper(attr_obj):
            def _fn(img, target_class_override=None):
                tc = target_class if target_class_override is None else int(target_class_override)
                # Attribution is computed w.r.t. full grid, but forward pass only processes target cell
                heat = attr_obj.attribute(img, target_class=tc)
                return heat
            return _fn

        method_results_map = {}
        for mname, attr_obj in methods.items():
            for k in args.k_percents:
                smoother.attribution_func = make_attr_wrapper(attr_obj)
                res = smoother.certify(
                    image,
                    k_percent=k,
                    target_class=target_class,
                    sigma=args.sigma,
                    num_samples=args.num_samples,
                    tau=args.tau,
                    batch_size=args.batch_size,
                    alpha=args.alpha,
                    save_noisy_samples=args.save_noisy_samples,
                    max_noisy_samples=args.max_noisy_samples,
                )
                
                results.setdefault(mname, {}).setdefault(k, []).append({
                    "image_idx": batch_idx,
                    "label": target_class,
                    "head_id": head_id,
                    "scale": scale,
                    "target_cell": grid_ds.target_cell,
                    "results": res,
                })
                
                # Save artifacts (certified maps + visualizations)
                save_artifacts(heatmap_dir, "resnet18", mname, k, batch_idx, image, res)
                
                # Collect for method_results_map
                method_results_map.setdefault(mname, {})[k] = res
        
        # Generate paper-style panel for first N images
        if panel_count < args.panel_examples:
            panel_path = generate_paper_panel_all_methods(
                heatmap_dir / "panels", "resnet18", batch_idx, image, method_results_map
            )
            print(f"  Generated paper panel: {panel_path}")
            panel_count += 1
        
        # Collect for Figure 4 style
        all_images_for_fig4.append({
            "image_tensor": image,
            "method_results_map": method_results_map,
        })

    # Generate Figure 4 style panel
    if all_images_for_fig4:
        fig4_path = generate_figure4_style_panel(heatmap_dir / "panels", "resnet18", all_images_for_fig4)
        print(f"Generated Figure 4 style panel: {fig4_path}")

    # Save pickle
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path = save_dir / f"results_{ts}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"resnet18": results}, f)
    print(f"Saved certification results to {pkl_path}")
    print(f"Saved certified maps and visualizations to {heatmap_dir}")


if __name__ == "__main__":
    main()
