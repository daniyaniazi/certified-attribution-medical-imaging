#!/usr/bin/env python
"""
Regenerate paper-style certification panels from saved certification results pickle.

- Loads the dataset (val split) to fetch original images.
- Reads cert_results from a pickle produced by certify_*_server.py.
- Rebuilds paper-style panels for every certified image of every model in the pickle.
- Does NOT rerun certification or smoothing.

Usage example:
  python regenerate_cert_panels.py \
    --results_pkl output/certify/isic/results_20251230_120000.pkl \
    --data_root data/raw/isic \
    --certify_dir output/certify/isic \
    --split val

Optional:
  --models resnet18 densenet121   # limit to these models
  --methods IntegratedGradients GradCAM RISE Occlusion LRP  # reorder/limit
  --max_images 0  # 0 means no limit
"""

import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Dataset loaders per project
from src.datasets.isic import ISICDataset
from src.datasets.chestxray import ChestXrayDataset
from src.datasets.brain_mri import BrainMRIDataset
from src.datasets.fundus import FundusDataset

DATASET_LOADERS = {
    "isic": ISICDataset,
    "chestxray": ChestXrayDataset,
    "brain_mri": BrainMRIDataset,
    "fundus": FundusDataset,
}

DEFAULT_METHODS = ["IntegratedGradients", "GradCAM", "RISE", "Occlusion", "LRP"]


def _tensor_to_hwc(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().clamp(0, 1).numpy()
    if img.ndim == 4:
        img = img[0]
    return np.transpose(img, (1, 2, 0))


def generate_paper_panel_all_methods(save_dir: Path, model_name: str, img_idx: int,
                                     img_tensor: torch.Tensor, method_results_map: dict,
                                     methods_order=None):
    if methods_order is None:
        methods_order = DEFAULT_METHODS
    save_dir.mkdir(parents=True, exist_ok=True)
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
        ax_input.text(-0.1, 0.5, mname, transform=ax_input.transAxes,
                     fontsize=11, fontweight='bold', va='center', ha='right', rotation=0)
        ax_input.axis("off")

        # Column 1: SS (Smoothed Sparsified)
        ax_ss = axes[r, 1]
        ss_map = None
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
                viz_map = np.ones((c_map.shape[0], c_map.shape[1], 3))
                viz_map[c_map == 0] = [1.0, 1.0, 1.0]
                viz_map[c_map == 1] = [1.0, 0.65, 0.0]
                viz_map[c_map == -1] = [0.85, 0.85, 0.85]
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
        overlay_map = np.ones(img.shape[:2] + (3,)) * 0.9
        overlay_map[c50 == 1] = [1.0, 0.65, 0.0]
        overlay_map[c25 == 1] = [0.9, 0.3, 0.3]
        overlay_map[c5 == 1] = [0.2, 0.0, 0.2]
        axo.imshow(overlay_map)
        if r == 0:
            axo.set_title('Overlayed', fontsize=12, fontweight='bold')
        axo.axis("off")

    fig.text(0.63, 0.97, 'Certified', ha='center', fontsize=13, fontweight='bold')
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


def build_image_index(cert_results: dict, methods_order):
    per_model = {}
    for model_name, methods_dict in cert_results.items():
        images_map = defaultdict(lambda: {"label": None, "method_results_map": defaultdict(dict)})
        for method_name, k_dict in methods_dict.items():
            if methods_order and method_name not in methods_order:
                continue
            for k_percent, entries in k_dict.items():
                for entry in entries:
                    img_idx = entry.get("image_idx")
                    label = entry.get("label")
                    res = entry.get("results", {})
                    images_map[img_idx]["label"] = label
                    images_map[img_idx]["method_results_map"][method_name][k_percent] = res
        per_model[model_name] = images_map
    return per_model


def load_dataset(dataset_name: str, data_root: Path, split: str):
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'")
    ds_cls = DATASET_LOADERS[dataset_name]
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return ds_cls(str(data_root), split=split, transform=val_transform)


def parse_args():
    p = argparse.ArgumentParser("Regenerate certification panels from pickle")
    p.add_argument("--results_pkl", required=True, help="Path to results_<timestamp>.pkl")
    p.add_argument("--certify_dir", required=True, help="Base certify dir (panels will be saved here)")
    p.add_argument("--data_root", required=True, help="Dataset root (e.g., data/raw/isic)")
    p.add_argument("--split", default="val")
    p.add_argument("--dataset", required=True, choices=list(DATASET_LOADERS.keys()))
    p.add_argument("--models", nargs="*", default=None, help="Optional whitelist of model names")
    p.add_argument("--methods", nargs="*", default=None, help="Optional whitelist/order of methods")
    p.add_argument("--max_images", type=int, default=0, help="Limit images per model (0 = no limit)")
    return p.parse_args()


def main():
    args = parse_args()
    results_pkl = Path(args.results_pkl)
    certify_dir = Path(args.certify_dir)
    data_root = Path(args.data_root)
    methods_order = args.methods if args.methods else DEFAULT_METHODS

    with open(results_pkl, "rb") as f:
        cert_results = pickle.load(f)

    per_model = build_image_index(cert_results, methods_order)
    dataset = load_dataset(args.dataset, data_root, args.split)

    for model_name, images_map in per_model.items():
        if args.models and model_name not in args.models:
            continue
        print(f"Model: {model_name} (images: {len(images_map)})")
        count = 0
        for img_idx, data in images_map.items():
            if args.max_images and count >= args.max_images:
                break
            method_results_map = {m: dict(kdict) for m, kdict in data["method_results_map"].items()}
            if not method_results_map:
                continue
            sample = dataset[img_idx]
            image_tensor = sample["image"] if isinstance(sample, dict) else sample[0]
            out_panel = generate_paper_panel_all_methods(certify_dir, model_name, img_idx, image_tensor, method_results_map, methods_order)
            print(f"  ✓ Saved panel: {out_panel}")
            count += 1

    print("Done. Panels regenerated from saved certification results.")


if __name__ == "__main__":
    main()
