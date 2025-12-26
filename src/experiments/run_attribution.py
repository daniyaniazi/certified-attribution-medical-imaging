"""
Attribution generation runner: compute attribution maps for test images.

Usage:
    python run_attribution.py --checkpoint outputs/checkpoints/best_model.pt --method integrated_gradients --dataset chexpert
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
from src.utils.io import save_attribution, load_checkpoint, save_json
from src.utils.viz import save_attribution_heatmap
from configs.defaults import DATASET_CONFIGS, ATTRIBUTION_METHODS


def get_dataset_loader(dataset_name: str, config: dict):
    """Get test dataset and DataLoader."""
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
        # Get target layer
        target_layer = model.layer4 if hasattr(model, 'layer4') else model.features
        return GradCAM(model, target_layer, device=device)
    elif method_name == 'rise':
        return RISE(model, device=device)
    elif method_name == 'occlusion':
        return Occlusion(model, device=device)
    else:
        raise ValueError(f"Unknown attribution method: {method_name}")


def main(args):
    """Main attribution generation script."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Recreate model
    # Extract model info from checkpoint path
    path_parts = args.checkpoint.split(os.sep)
    dataset_name = path_parts[-2] if len(path_parts) > 1 else 'chexpert'
    model_name = path_parts[-1].split('_')[0] if len(path_parts) > 0 else 'resnet18'
    
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
    
    # Create output directory
    output_dir = os.path.join(
        'outputs/attributions_raw',
        args.dataset,
        args.model,
        args.method
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate attributions
    attributions_list = []
    
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Generating attributions")
        
        for idx, batch in pbar:
            image = batch['image'].to(device)  # [1, C, H, W]
            label = batch['label'].item()
            meta = batch['meta']
            
            # Skip if not positive class (optional)
            if args.positive_only and label == 0:
                continue
            
            # Compute attribution
            if args.method == 'integrated_gradients':
                attr = attr_method.attribute(image, target_class=1, num_steps=50)
            elif args.method == 'gradcam':
                attr = attr_method.attribute(image, target_class=1)
            elif args.method == 'rise':
                attr = attr_method.attribute(image, target_class=1, num_samples=500)
            elif args.method == 'occlusion':
                attr = attr_method.attribute(image, target_class=1, patch_size=16)
            
            # Save attribution array
            sample_id = meta['id'][0] if isinstance(meta['id'], list) else meta['id']
            attr_path = os.path.join(output_dir, f'{sample_id}_attr.npy')
            save_attribution(attr, attr_path)
            
            # Optionally save visualization
            if args.save_viz:
                img_np = image[0].permute(1, 2, 0).cpu().numpy()
                viz_path = os.path.join(output_dir, f'{sample_id}_viz.png')
                save_attribution_heatmap(img_np, attr, viz_path)
            
            attributions_list.append({
                'id': str(sample_id),
                'label': int(label),
                'filename': meta['filename'][0] if isinstance(meta['filename'], list) else meta['filename']
            })
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.json')
    save_json({
        'method': args.method,
        'model': args.model,
        'dataset': args.dataset,
        'num_samples': len(attributions_list),
        'samples': attributions_list
    }, metadata_path)
    
    print(f"\n✓ Generated {len(attributions_list)} attributions")
    print(f"✓ Saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate attribution maps')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='chexpert',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model name')
    parser.add_argument('--method', type=str, default='integrated_gradients',
                        choices=['integrated_gradients', 'gradcam', 'rise', 'occlusion'],
                        help='Attribution method')
    parser.add_argument('--save-viz', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--positive-only', action='store_true',
                        help='Only process positive samples')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Limit number of samples (for testing)')
    
    args = parser.parse_args()
    main(args)
