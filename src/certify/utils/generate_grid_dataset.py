"""Generate grid-based datasets for CertifiedGridPG evaluation.

Builds synthetic grid images (e.g., 2x2) from existing datasets.
Each grid cell contains an image from a chosen class; metadata records
which class is in which cell so localization ground truth is known.
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.isic import ISICDataset
from src.datasets.chestxray import ChestXrayDataset
from src.datasets.brain_mri import BrainMRIDataset
from src.datasets.fundus import FundusDataset
from src.datasets.base import BaseDataset

DATASET_REGISTRY = {
    "isic": ISICDataset,
    "chestxray": ChestXrayDataset,
    "brain_mri": BrainMRIDataset,
    "fundus": FundusDataset,
}


def _load_dataset(name: str, root: Path, split: str, target_size: Tuple[int, int]) -> BaseDataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset: {name}")
    transform = BaseDataset.get_default_transform(target_size=target_size)
    return DATASET_REGISTRY[name](root_dir=str(root), split=split, transform=transform)


def _build_class_pools(dataset: BaseDataset) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    pools: Dict[int, List[int]] = {}
    for idx, sample in enumerate(dataset.samples):
        pools.setdefault(sample["label"], []).append(idx)
    # Keep originals for refills when sampling with replacement
    originals = {k: v.copy() for k, v in pools.items()}
    for v in pools.values():
        random.shuffle(v)
    return pools, originals


def _pop_from_pool(label: int, pools: Dict[int, List[int]], originals: Dict[int, List[int]]) -> int:
    if not pools[label]:
        pools[label] = originals[label].copy()
        random.shuffle(pools[label])
    return pools[label].pop()


def _grid_coordinates(scale: int, cell_idx: int, cell_h: int, cell_w: int) -> Tuple[int, int, int, int]:
    row = cell_idx // scale
    col = cell_idx % scale
    y = row * cell_h
    x = col * cell_w
    return y, x, cell_h, cell_w


def generate_grids(
    dataset: BaseDataset,
    num_grids: int,
    scale: int = 2,
    target_cell: int = 0,
    allow_repeats: bool = False,
) -> Dict:
    assert scale >= 1
    assert target_cell < scale * scale

    pools, originals = _build_class_pools(dataset)
    class_ids = sorted(pools.keys())
    num_classes = len(class_ids)

    # If classes are fewer than grid cells, force repeats
    if num_classes < scale * scale:
        allow_repeats = True

    # Peek one sample to get shape
    sample = dataset[0]
    img_c, img_h, img_w = sample["image"].shape

    grids = torch.zeros((num_grids, img_c, img_h * scale, img_w * scale))
    cell_classes = torch.zeros((num_grids, scale * scale), dtype=torch.long)
    source_indices = torch.zeros((num_grids, scale * scale), dtype=torch.long)
    target_classes = torch.zeros((num_grids,), dtype=torch.long)

    for grid_idx in range(num_grids):
        if allow_repeats:
            chosen = [random.choice(class_ids) for _ in range(scale * scale)]
        else:
            chosen = random.sample(class_ids, k=scale * scale)

        cell_classes[grid_idx] = torch.tensor(chosen)
        target_classes[grid_idx] = chosen[target_cell]

        for cell_idx, label in enumerate(chosen):
            ds_idx = _pop_from_pool(label, pools, originals)
            item = dataset[ds_idx]
            img = item["image"]
            if img.shape[1:] != (img_h, img_w):
                raise ValueError("All images must share the same spatial size after transforms")
            y, x, h, w = _grid_coordinates(scale, cell_idx, img_h, img_w)
            grids[grid_idx, :, y:y + h, x:x + w] = img
            source_indices[grid_idx, cell_idx] = ds_idx

    class_names = getattr(dataset, "classes", None)

    return {
        "images": grids,
        "cell_classes": cell_classes,
        "source_indices": source_indices,
        "target_classes": target_classes,
        "target_cell": target_cell,
        "scale": scale,
        "image_size": (img_h, img_w),
        "dataset": dataset.__class__.__name__,
        "split": dataset.split,
        "num_classes": num_classes,
        "class_names": class_names,
    }


def save_grid_dataset(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def build_grid_masks(payload: Dict) -> Dict[int, np.ndarray]:
    """
    Build binary masks for each grid image's target cell.
    
    Args:
        payload: dict from generate_grids() with target_cell, scale, image_size
    
    Returns:
        {grid_image_idx -> binary mask [H*scale, W*scale]}
    
    Example (2×2 grid, 224×224 tiles, target_cell=0):
        - Mask is 448×448 with ones in top-left 224×224 and zeros elsewhere.
    """
    scale = payload['scale']
    target_cell = payload['target_cell']
    img_h, img_w = payload['image_size']
    num_grids = payload['images'].shape[0]
    
    masks = {}
    
    for grid_idx in range(num_grids):
        # Create a binary mask for the target cell
        full_h = img_h * scale
        full_w = img_w * scale
        mask = np.zeros((full_h, full_w), dtype=np.uint8)
        
        # Compute target cell position
        target_row = target_cell // scale
        target_col = target_cell % scale
        y = target_row * img_h
        x = target_col * img_w
        
        # Set target cell region to 1
        mask[y:y+img_h, x:x+img_w] = 1
        
        masks[grid_idx] = mask
    
    return masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grid dataset for CertifiedGridPG")
    parser.add_argument("--dataset", required=True, choices=DATASET_REGISTRY.keys(), help="Dataset name")
    parser.add_argument("--data_root", default="data/raw", help="Root directory containing dataset splits")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num_grids", type=int, default=100, help="Number of grid images to generate")
    parser.add_argument("--scale", type=int, default=2, help="Grid dimension (n for n x n grid)")
    parser.add_argument("--target_cell", type=int, default=0, help="Index of cell treated as target (0 = top-left)")
    parser.add_argument("--target_size", type=int, default=224, help="Base image size before tiling")
    parser.add_argument("--allow_repeats", action="store_true", help="Allow repeating classes within a grid")
    parser.add_argument(
        "--output", 
        default=None,
        help="Path to save generated grid dataset (.pt). Defaults to data/processed/[dataset]_grid_{scale}x{scale}/{split}/grid.pt"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = _load_dataset(
        name=args.dataset,
        root=Path(args.data_root) / args.dataset,
        split=args.split,
        target_size=(args.target_size, args.target_size),
    )

    payload = generate_grids(
        dataset=dataset,
        num_grids=args.num_grids,
        scale=args.scale,
        target_cell=args.target_cell,
        allow_repeats=args.allow_repeats,
    )

    if args.output is None:
        out_dir = Path("data/processed") / f"{args.dataset}_grid_{args.scale}x{args.scale}" / args.split
        output_path = out_dir / "grid.pt"
    else:
        output_path = Path(args.output)

    save_grid_dataset(payload, output_path)
    print(f"✓ Saved grid dataset to {output_path}")
    print(f"  Images: {payload['images'].shape[0]}")
    print(f"  Grid size: {args.scale}x{args.scale} (cell {args.target_cell} is target)")
    print(f"  Classes used: {payload['num_classes']}")


if __name__ == "__main__":
    main()
