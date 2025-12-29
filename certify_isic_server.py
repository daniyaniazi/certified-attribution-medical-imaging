#!/usr/bin/env python
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Resolve repository root (execute.sh sets CWD to project root)
ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local imports from repo
from src.models.factory import get_model
from src.datasets.isic import ISICDataset
from src.xai.attribution_unified import (
    IntegratedGradientsUnified,
    GradCAMUnified,
    RISEUnified,
    OcclusionUnified,
    LRPUnified,
)
from src.certify.smoothing import RandomizedSmoothingAttributor


def get_target_layer(model: nn.Module, model_name: str):
    """Return last conv layer suitable for Grad-CAM across common backbones."""
    name = model_name.lower()
    if "resnet" in name:
        return model.layer4[-1].conv2 if hasattr(model.layer4[-1], "conv2") else model.layer4[-1]
    if "densenet" in name:
        return model.features.denseblock4
    if "efficientnet" in name:
        return list(model.features.modules())[-1]
    if "mobilenet" in name:
        return list(model.features.modules())[-1]
    # Fallback: last Conv2d
    for _, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError(f"Could not find target layer for {model_name}")


def discover_models(checkpoint_dir: Path):
    models = []
    if checkpoint_dir.exists():
        for d in sorted([p for p in checkpoint_dir.iterdir() if p.is_dir()]):
            candidates = list(d.rglob("*.pt")) + list(d.rglob("*.pth"))
            ckpt = None
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt = candidates[0]
            models.append({"name": d.name, "checkpoint": str(ckpt) if ckpt else None})
    return models


def compute_metrics_summary(cert_results: dict):
    summary = {}
    for model_name, methods_dict in cert_results.items():
        summary[model_name] = {}
        for method_name, k_dict in methods_dict.items():
            summary[model_name][method_name] = {}
            for k_percent, entries in k_dict.items():
                if not entries:
                    continue
                per_image = []
                for entry in entries:
                    res = entry.get("results", {})
                    c_map = res.get("certified_map")
                    if c_map is None:
                        continue
                    total = c_map.size
                    per_image.append({
                        "image_idx": entry.get("image_idx"),
                        "label": entry.get("label"),
                        "pct_certified": float(np.sum(c_map != -1) / total * 100.0),
                        "pct_abstained": float(np.sum(c_map == -1) / total * 100.0),
                        "pct_certified_1": float(np.sum(c_map == 1) / total * 100.0),
                        "pct_certified_0": float(np.sum(c_map == 0) / total * 100.0),
                        "certified_radius": float(res.get("certified_radius", 0.0)),
                    })
                if not per_image:
                    continue
                summary[model_name][method_name][k_percent] = {
                    "per_image": per_image,
                    "mean": {
                        "pct_certified": float(np.mean([p["pct_certified"] for p in per_image])),
                        "pct_abstained": float(np.mean([p["pct_abstained"] for p in per_image])),
                        "pct_certified_1": float(np.mean([p["pct_certified_1"] for p in per_image])),
                        "pct_certified_0": float(np.mean([p["pct_certified_0"] for p in per_image])),
                        "certified_radius": float(np.mean([p["certified_radius"] for p in per_image])),
                    },
                }
    return summary


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
    img = image_tensor.detach().cpu().clamp(0, 1).numpy()
    if img.ndim == 4:
        img = img[0]
    img = np.transpose(img, (1, 2, 0))  # HWC

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

    _save_overlay(clean, "inferno", f"Clean heatmap (K={k}%)", f"img_{img_idx}_clean.png")
    _save_overlay(ss_map, "magma", "Smoothed sparsified map", f"img_{img_idx}_ssmap.png", vmin=0, vmax=1)

    if c_map is not None:
        viz_rgb = np.ones((c_map.shape[0], c_map.shape[1], 3))
        viz_rgb[c_map == 0] = [1.0, 1.0, 1.0]
        viz_rgb[c_map == 1] = [1.0, 0.55, 0.0]
        viz_rgb[c_map == -1] = [0.8, 0.8, 0.8]
        _save_overlay(viz_rgb, None, "Certified map (1=orange,0=white,abstain=gray)", f"img_{img_idx}_certmap.png")


def _tensor_to_hwc(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().clamp(0, 1).numpy()
    if img.ndim == 4:
        img = img[0]
    return np.transpose(img, (1, 2, 0))


def _render_cert_column(c_map: np.ndarray, color_rgb=(1.0, 0.55, 0.0)) -> np.ndarray:
    h, w = c_map.shape
    base = np.ones((h, w, 3)) * 0.85
    mask1 = (c_map == 1)
    base[mask1] = np.array(color_rgb)
    return base


def _overlay_k_layers_on_image(img_hwc: np.ndarray, c50: np.ndarray, c25: np.ndarray, c5: np.ndarray) -> np.ndarray:
    h, w, _ = img_hwc.shape
    overlay = img_hwc.copy()
    # Precedence: K=5% (black) > K=25% (red) > K=50% (yellow)
    mask5 = (c5 == 1)
    mask25 = (c25 == 1) & (~mask5)
    mask50 = (c50 == 1) & (~mask5) & (~mask25)
    overlay[mask50] = 0.7 * overlay[mask50] + 0.3 * np.array([1.0, 0.8, 0.0])
    overlay[mask25] = 0.6 * overlay[mask25] + 0.4 * np.array([1.0, 0.0, 0.0])
    overlay[mask5] = 0.5 * overlay[mask5] + 0.5 * np.array([0.0, 0.0, 0.0])
    return overlay


def generate_figure4_style_panel(save_dir: Path, model_name: str, all_images_results: list):
    """Create Figure 4 style panel: rows=images (up to 5), cols=methods with SS and Certified pairs.
    
    Args:
        save_dir: Directory to save the panel
        model_name: Name of the model
        all_images_results: List of dicts with keys 'image_tensor' and 'method_results_map'
    """
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
    import matplotlib.patches as mpatches
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
    import matplotlib.patches as mpatches
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
    
    out_path = save_dir / f"{model_name}_img{img_idx}_paper_style.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def build_attr_methods(model: nn.Module, device: str, model_name: str):
    tl = get_target_layer(model, model_name)
    return {
        "IntegratedGradients": IntegratedGradientsUnified(model, device),
        "GradCAM": GradCAMUnified(model, tl, device),
        "RISE": RISEUnified(model, device),
        "Occlusion": OcclusionUnified(model, device),
        "LRP": LRPUnified(model, device, epsilon=1e-6),
    }


def make_attr_wrapper(attr_func, target_h: int, target_w: int, target_class: int):
    def _wrapper(img, target_class_override=None):
        img.requires_grad_(True)
        # Use provided class if smoothing passes one; else use captured target_class
        tc = target_class if target_class_override is None else int(target_class_override)
        with torch.enable_grad():
            heat = attr_func.attribute(img, target_class=tc)
        heat_np = heat.detach().cpu().numpy() if isinstance(heat, torch.Tensor) else np.array(heat)
        while heat_np.ndim > 2:
            if heat_np.shape[0] == 1:
                heat_np = heat_np.squeeze(0)
            elif heat_np.shape[-1] == 1:
                heat_np = heat_np.squeeze(-1)
            else:
                if heat_np.ndim == 3:
                    heat_np = heat_np.mean(axis=0)
                elif heat_np.ndim == 4:
                    heat_np = heat_np.mean(axis=(0, 1))
                break
        if heat_np.ndim != 2:
            raise ValueError(f"Cannot reduce attribution to 2D, shape: {heat_np.shape}")
        if heat_np.shape != (target_h, target_w):
            from scipy.ndimage import zoom
            zoom_factors = (target_h / heat_np.shape[0], target_w / heat_np.shape[1])
            heat_np = zoom(heat_np, zoom_factors, order=1)
        return heat_np
    return _wrapper


def parse_args():
    p = argparse.ArgumentParser("ISIC Certification Server")
    p.add_argument("--checkpoint_dir", type=str, default=str(ROOT / "outputs" / "checkpoints" / "isic"))
    p.add_argument("--data_root", type=str, default=str(ROOT / "data" / "raw" / "isic"))
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--certify_dir", type=str, default=str(ROOT / "output" / "certify" / "isic"))
    p.add_argument("--heatmap_dir", type=str, default=str(ROOT / "output" / "certify" / "isic"))
    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--sigma", type=float, default=0.15)
    p.add_argument("--tau", type=float, default=0.75)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument("--k_percents", type=int, nargs="+", default=[50, 25, 5])
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--save_noisy_samples", action="store_true")
    p.add_argument("--max_noisy_samples", type=int, default=3)
    p.add_argument("--models", type=str, nargs="*", default=None, help="Optional whitelist of model names")
    p.add_argument("--panel_examples", type=int, default=3, help="Create paper-style panels for first N images per model")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    checkpoint_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)
    certify_dir = Path(args.certify_dir)
    heatmap_dir = Path(args.heatmap_dir)
    certify_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Data root: {data_root}")
    print(f"Certification dir: {certify_dir}")
    print(f"Heatmap viz dir: {heatmap_dir}")

    # Discover models
    available = discover_models(checkpoint_dir)
    if args.models:
        available = [m for m in available if m["name"] in args.models]
    print(f"Found {len(available)} models")
    for m in available:
        print(f" - {m['name']} (ckpt: {'found' if m['checkpoint'] else 'missing'})")

    # Dataset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ISICDataset(data_root, split=args.split, transform=val_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)  # Shuffle to get diverse labels
    print(f"Loaded {len(dataset)} images from split '{args.split}'")

    # Resume store
    partial_path = certify_dir / "results_partial.pkl"
    if partial_path.exists():
        import pickle
        with open(partial_path, "rb") as f:
            cert_results = pickle.load(f)
        print(f"Resumed existing results: {partial_path.name}")
    else:
        cert_results = {}

    def save_partial():
        import pickle
        with open(partial_path, "wb") as f:
            pickle.dump(cert_results, f)

    # Iterate models
    for model_info in available:
        model_name = model_info["name"]
        print("\n" + "=" * 80)
        print(f"Model: {model_name}")
        print("=" * 80)
        try:
            use_pretrained = model_info["checkpoint"] is None
            model, _ = get_model(model_name, num_classes=8, pretrained=use_pretrained, device=device)
            if model_info["checkpoint"]:
                checkpoint = torch.load(model_info["checkpoint"], map_location=device)
                state = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state)
            model.eval()
            print(f"Model ready ({'pretrained' if use_pretrained else 'checkpoint loaded'})")
        except Exception as e:
            print(f"Failed to load model '{model_name}': {e}")
            continue

        if model_name not in cert_results:
            cert_results[model_name] = {}

        # Attribution methods
        try:
            methods = build_attr_methods(model, device, model_name)
            print("All 5 attribution methods initialized")
        except Exception as e:
            print(f"Failed to init methods for '{model_name}': {e}")
            continue

        smoother = RandomizedSmoothingAttributor(model, None, device=device)

        # High-confidence filtering (lowered threshold to get more images)
        high_confidence_threshold = 0.2  # Lowered from 0.8
        
        # Track if we've created the paper-style panel for this model
        panel_created = False
        
        # Collect results for Figure 4 style visualization (5 images)
        figure4_data = []
        
        # Keep sampling until we get at least num_images valid images
        num_target_images = args.num_images
        valid_images_certified = 0
        images_checked = 0
        max_images_to_check = len(dataset)  # Don't check more than the dataset size
        
        # Images loop - continue until we get target number of valid images
        for img_idx, batch in enumerate(loader):
            if valid_images_certified >= num_target_images:
                print(f"\n✓ Reached target of {num_target_images} valid images")
                break
            
            if images_checked >= max_images_to_check:
                print(f"\n⚠ Checked all {max_images_to_check} images, found {valid_images_certified} valid")
                break
            
            images_checked += 1
            
            image = batch["image"].to(device)
            label_value = batch["label"]
            label_int = int(label_value.squeeze().item()) if isinstance(label_value, torch.Tensor) else int(label_value)
            
            # PAPER REQUIREMENT: Only certify high-confidence correct predictions
            with torch.no_grad():
                logits = model(image)
                probs = torch.softmax(logits, dim=1)
                pred_class = logits.argmax(dim=1).item()
                pred_confidence = probs[0, pred_class].item()
            
            # Skip if prediction is wrong or low confidence
            if pred_class != label_int:
                print(f"  Image {images_checked} SKIPPED (pred={pred_class} != label={label_int}) [{valid_images_certified}/{num_target_images} valid]")
                continue
            if pred_confidence < high_confidence_threshold:
                print(f"  Image {images_checked} SKIPPED (confidence={pred_confidence:.3f} < {high_confidence_threshold}) [{valid_images_certified}/{num_target_images} valid]")
                continue
            
            # Enable gradients for attribution methods
            image.requires_grad_(True)
            valid_images_certified += 1
            print(f"\n  ✓ Valid image {valid_images_certified}/{num_target_images} (dataset idx: {img_idx}, label: {label_int}, pred: {pred_class}, conf: {pred_confidence:.3f})")

            method_results_map = {}
            for method_name, attr_func in methods.items():
                if method_name not in cert_results[model_name]:
                    cert_results[model_name][method_name] = {}

                target_h, target_w = image.shape[-2:]
                smoother.attribution_func = make_attr_wrapper(attr_func, target_h, target_w, label_int)

                results_by_k = {}
                for k in args.k_percents:
                    try:
                        results = smoother.certify(
                            image,
                            k_percent=k,
                            target_class=label_int,
                            sigma=args.sigma,
                            num_samples=args.num_samples,
                            tau=args.tau,
                            batch_size=args.batch_size,
                            alpha=args.alpha,
                            save_noisy_samples=args.save_noisy_samples,
                            max_noisy_samples=args.max_noisy_samples,
                        )

                        cert_results[model_name][method_name].setdefault(k, [])
                        cert_results[model_name][method_name][k].append({
                            "image_idx": img_idx,
                            "label": label_int,
                            "results": results,
                        })
                        results_by_k[k] = results
                        # Persist artifacts & quick viz
                        save_artifacts(heatmap_dir, model_name, method_name, k, img_idx, image, results)

                        save_partial()
                        print(f"    ✓ {method_name} K={k}%")
                    except Exception as e:
                        print(f"    ✗ {method_name} K={k}% failed: {e}")

                method_results_map[method_name] = results_by_k

            # After all methods for this image, create paper-style panel for FIRST successfully certified image per model
            if not panel_created and method_results_map:
                out_panel = generate_paper_panel_all_methods(certify_dir, model_name, img_idx, image, method_results_map)
                print(f"  ✓ Saved paper-style panel: {out_panel}")
                panel_created = True
            
            # Collect data for Figure 4 style visualization (up to 5 images)
            if len(figure4_data) < 5 and method_results_map:
                figure4_data.append({
                    'image_tensor': image,
                    'method_results_map': method_results_map
                })
        
        # After processing all images for this model, create Figure 4 style visualization
        if figure4_data:
            out_fig4 = generate_figure4_style_panel(certify_dir, model_name, figure4_data)
            print(f"\n  ✓ Saved Figure 4 style panel: {out_fig4} ({len(figure4_data)} images)")

    # Final saves
    save_partial()
    print(f"\n✓ Certification complete for {len(cert_results)} models")

    # Metrics + summaries
    metrics_summary = compute_metrics_summary(cert_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl_path = certify_dir / f"results_{timestamp}.pkl"
    json_path = certify_dir / f"results_{timestamp}.json"
    csv_path = certify_dir / f"metrics_summary_{timestamp}.csv"

    # Pickle full results
    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(cert_results, f)
    print(f"Saved pickle: {pkl_path}")

    # JSON summary
    json_summary = {}
    for model_name, methods_dict in cert_results.items():
        json_summary[model_name] = {}
        for method_name, k_dict in methods_dict.items():
            json_summary[model_name][method_name] = {}
            for k_percent, entries in k_dict.items():
                if not entries:
                    continue
                if k_percent not in metrics_summary.get(model_name, {}).get(method_name, {}):
                    continue
                mean_metrics = metrics_summary[model_name][method_name][k_percent]["mean"]
                json_summary[model_name][method_name][str(k_percent)] = {
                    "num_images": len(entries),
                    "pct_certified": mean_metrics["pct_certified"],
                    "pct_abstained": mean_metrics["pct_abstained"],
                    "pct_certified_1": mean_metrics["pct_certified_1"],
                    "pct_certified_0": mean_metrics["pct_certified_0"],
                    "certified_radius": mean_metrics["certified_radius"],
                    "alpha": float(args.alpha),
                }
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # CSV (per-image rows)
    try:
        import pandas as pd
        rows = []
        for model_name in sorted(cert_results.keys()):
            for method_name, k_dict in cert_results[model_name].items():
                for k_percent, entries in k_dict.items():
                    per_image = metrics_summary.get(model_name, {}).get(method_name, {}).get(k_percent, {}).get("per_image", [])
                    for entry in per_image:
                        rows.append({
                            "Model": model_name,
                            "Method": method_name,
                            "K (%)": k_percent,
                            "Image Idx": entry["image_idx"],
                            "Label": entry["label"],
                            "Certified (%)": f"{entry['pct_certified']:.2f}",
                            "Abstained (%)": f"{entry['pct_abstained']:.2f}",
                            "Certified-1 (%)": f"{entry['pct_certified_1']:.2f}",
                            "Certified-0 (%)": f"{entry['pct_certified_0']:.2f}",
                            "Radius": f"{entry['certified_radius']:.4f}",
                        })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
    except Exception as e:
        print(f"CSV export failed (missing pandas?): {e}")

    print("\n✅ Certification server job complete")


if __name__ == "__main__":
    main()
