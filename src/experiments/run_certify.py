"""
Certification runner: compute certified attributions with randomized smoothing.

Usage:
    python run_certify.py --checkpoint outputs/checkpoints/best_model.pt --method integrated_gradients
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.factory import get_model
from src.datasets.chexpert import CheXpertDataset
from src.datasets.isic import ISICDataset
from src.xai.attribution import IntegratedGradients, GradCAM, RISE, Occlusion
from src.certify.sparsify import sparsify_topk
from src.certify.smoothing import RandomizedSmoothingAttributor
from src.certify.evaluate import CertificationEvaluator
from src.utils.io import save_attribution, load_checkpoint, save_json
from src.utils.viz import save_certified_map
from configs.defaults import DATASET_CONFIGS, CERTIFICATION_CONFIG


def get_dataset_loader(dataset_name: str, config: dict):
    """Get test dataset."""
    if dataset_name == 'chexpert':
        dataset = CheXpertDataset(
            root_dir=config.get('root_dir', 'data/raw/chexpert'),
            split='test',
            task=config.get('task', 'pneumonia'),
            target_size=config['target_size']
        )
    elif dataset_name == 'isic':
        dataset = ISICDataset(
            root_dir=config.get('root_dir', 'data/raw/isic'),
            split='test',
            target_size=config['target_size']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return dataset, loader


def get_attribution_method(method_name: str, model, device: str):
    """Get attribution method instance."""
    if method_name == 'integrated_gradients':
        return IntegratedGradients(model, device=device)
    elif method_name == 'gradcam':
        target_layer = model.layer4 if hasattr(model, 'layer4') else model.features
        return GradCAM(model, target_layer, device=device)
    elif method_name == 'rise':
        return RISE(model, device=device)
    elif method_name == 'occlusion':
        return Occlusion(model, device=device)
    else:
        raise ValueError(f"Unknown attribution method: {method_name}")


def main(args):
    """Main certification script."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Extract model info
    path_parts = args.checkpoint.split(os.sep)
    dataset_name = path_parts[-2] if len(path_parts) > 1 else 'chexpert'
    
    dataset_config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS['chexpert'])
    model, config = get_model(
        args.model,
        num_classes=dataset_config['num_classes'],
        pretrained=False,
        device=device
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load dataset
    print(f"Loading {args.dataset} test set...")
    dataset, loader = get_dataset_loader(args.dataset, dataset_config)
    print(f"Test samples: {len(dataset)}")
    
    # Create attribution method
    print(f"Using attribution method: {args.method}")
    attr_method = get_attribution_method(args.method, model, device)
    
    # Create smoother (Paper: Eq. 5-7)
    print("Creating randomized smoother...")
    smoother = RandomizedSmoothingAttributor(
        model,
        attr_method.attribute,
        device=device
    )
    
    # Create output directory
    output_dir = os.path.join(
        'outputs/attributions_certified',
        args.dataset,
        args.model,
        args.method,
        f'sigma{args.sigma}_tau{args.tau}_n{args.num_samples}'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Certification parameters (from paper)
    cert_params = {
        'sigma': args.sigma,
        'tau': args.tau,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'k_percents': [int(k) for k in args.k_percents.split(',')]
    }
    
    # Helper: compose m x m grid (default 2x2) from list of images [1,C,H,W]
    def compose_grid(images, m=2):
        assert len(images) == m * m
        # Assume all images same shape [1, C, H, W]
        _, C, H, W = images[0].shape
        rows = []
        bboxes = []  # (y0,x0,y1,x1) per cell
        for r in range(m):
            row_imgs = []
            for c in range(m):
                idx = r * m + c
                row_imgs.append(images[idx])
                y0, x0 = r * H, c * W
                y1, x1 = y0 + H, x0 + W
                bboxes.append((y0, x0, y1, x1))
            rows.append(torch.cat(row_imgs, dim=3))  # concat width
        grid = torch.cat(rows, dim=2)  # concat height
        return grid, bboxes

    # Process: either per-sample certification or GridPG grids
    evaluator = CertificationEvaluator()
    results_list = []

    with torch.no_grad():
        if args.gridpg:
            # Build and certify grids
            print(f"Generating {args.num_grids} grids of size {args.grid_size}x{args.grid_size}...")
            dataset_iter = iter(loader)
            grids_done = 0
            target_row, target_col = map(int, args.grid_target_cell.split(','))
            while grids_done < args.num_grids:
                images = []
                labels = []
                metas = []
                # Collect m*m images (optionally distinct labels)
                while len(images) < args.grid_size * args.grid_size:
                    try:
                        batch = next(dataset_iter)
                    except StopIteration:
                        dataset_iter = iter(loader)
                        batch = next(dataset_iter)
                    img = batch['image'].to(device)
                    images.append(img)
                    labels.append(batch['label'].item())
                    metas.append(batch['meta'])
                # Compose grid
                grid_img, bboxes = compose_grid(images, m=args.grid_size)  # [1,C,H*m,W*m]
                grid_id = f'grid_{grids_done:03d}'
                
                sample_results = {
                    'id': grid_id,
                    'grid_size': args.grid_size,
                    'labels': [int(l) for l in labels],
                    'target_cell': [target_row, target_col],
                    'certified_maps': []
                }

                # Certify for each K value
                for k_percent in cert_params['k_percents']:
                    results = smoother.certify(
                        grid_img,
                        k_percent=k_percent,
                        target_class=1,
                        sigma=args.sigma,
                        num_samples=args.num_samples,
                        tau=args.tau,
                        batch_size=args.batch_size,
                        alpha=args.alpha
                    )
                    certified_map = results['certified_map']
                    radius = results['certified_radius']
                    # GridPG via bbox of target cell
                    H = images[0].shape[2]
                    W = images[0].shape[3]
                    y0 = target_row * H
                    x0 = target_col * W
                    y1 = y0 + H
                    x1 = x0 + W
                    gridpg = evaluator.compute_certified_gridpg_bbox(
                        certified_map, y0, x0, y1, x1
                    )
                    cert_metrics = evaluator.evaluate_certified(results)
                    # Save arrays
                    certified_path = os.path.join(output_dir, f'{grid_id}_certified_k{k_percent}.npy')
                    save_attribution(certified_map.astype(np.float32), certified_path)
                    viz_path = os.path.join(output_dir, f'{grid_id}_certified_k{k_percent}.png')
                    save_certified_map(certified_map, viz_path, title=f'Grid {args.grid_size}x{args.grid_size} K={k_percent}% R={radius:.4f}')
                    sample_results['certified_maps'].append({
                        'k_percent': k_percent,
                        'pct_certified': cert_metrics['pct_certified'],
                        'pct_abstained': cert_metrics['pct_abstained'],
                        'certified_radius': radius,
                        'gridpg': gridpg,
                        'metrics': cert_metrics
                    })
                results_list.append(sample_results)
                grids_done += 1
        else:
            # Per-sample certification
            pbar = tqdm(enumerate(loader), total=len(loader), desc="Certifying")
            for idx, batch in pbar:
                image = batch['image'].to(device)  # [1, C, H, W]
                label = batch['label'].item()
                meta = batch['meta']
                sample_id = meta['id'][0] if isinstance(meta['id'], list) else meta['id']
                sample_results = {
                    'id': str(sample_id),
                    'label': int(label),
                    'filename': meta['filename'][0] if isinstance(meta['filename'], list) else meta['filename'],
                    'certified_maps': []
                }
                for k_percent in cert_params['k_percents']:
                    results = smoother.certify(
                        image,
                        k_percent=k_percent,
                        target_class=1,
                        sigma=args.sigma,
                        num_samples=args.num_samples,
                        tau=args.tau,
                        batch_size=args.batch_size,
                        alpha=args.alpha
                    )
                    certified_map = results['certified_map']
                    p_1 = results['p_1']
                    p_0 = results['p_0']
                    radius = results['certified_radius']
                    cert_metrics = evaluator.evaluate_certified(results)
                    certified_path = os.path.join(output_dir, f'{sample_id}_certified_k{k_percent}.npy')
                    save_attribution(certified_map.astype(np.float32), certified_path)
                    p1_path = os.path.join(output_dir, f'{sample_id}_p1_k{k_percent}.npy')
                    p0_path = os.path.join(output_dir, f'{sample_id}_p0_k{k_percent}.npy')
                    save_attribution(p_1.astype(np.float32), p1_path)
                    save_attribution(p_0.astype(np.float32), p0_path)
                    viz_path = os.path.join(output_dir, f'{sample_id}_certified_k{k_percent}.png')
                    save_certified_map(certified_map, viz_path, title=f'K={k_percent}% R={radius:.4f}')
                    sample_results['certified_maps'].append({
                        'k_percent': k_percent,
                        'pct_certified': cert_metrics['pct_certified'],
                        'pct_abstained': cert_metrics['pct_abstained'],
                        'certified_radius': radius,
                        'metrics': cert_metrics
                    })
                    pbar.set_postfix({'k%': k_percent, 'certified%': f'{cert_metrics["pct_certified"]:.1f}', 'R': f'{radius:.4f}'})
                results_list.append(sample_results)
    
    # Save results
    results_path = os.path.join(output_dir, 'results.json')
    save_json({
        'method': args.method,
        'model': args.model,
        'dataset': args.dataset,
        'cert_params': cert_params,
        'num_samples': len(results_list),
        'samples': results_list
    }, results_path)
    
    print(f"\n✓ Certified {len(results_list)} samples")
    print(f"✓ Saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Certify attributions with randomized smoothing')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='chexpert',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model name')
    parser.add_argument('--method', type=str, default='integrated_gradients',
                        choices=['integrated_gradients', 'gradcam', 'rise', 'occlusion'],
                        help='Attribution method')
    parser.add_argument('--sigma', type=float, default=0.15,
                        help='Gaussian noise std')
    parser.add_argument('--tau', type=float, default=0.75,
                        help='Certification threshold')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of smoothing samples')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for smoothing')
    parser.add_argument('--k-percents', type=str, default='50,30,10',
                        help='K percentiles for sparsification (comma-separated)')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Significance level for Clopper-Pearson confidence bounds')
    parser.add_argument('--gridpg', action='store_true',
                        help='Evaluate Certified GridPG by composing grids of images')
    parser.add_argument('--grid-size', type=int, default=2,
                        help='Grid dimension m (m x m)')
    parser.add_argument('--num-grids', type=int, default=100,
                        help='Number of grids to compose and certify')
    parser.add_argument('--grid-target-cell', type=str, default='0,0',
                        help='Target cell as row,col for GridPG (e.g., "0,1")')
    parser.add_argument('--num-test-samples', type=int, default=None,
                        help='Limit number of test samples')
    
    args = parser.parse_args()
    main(args)
