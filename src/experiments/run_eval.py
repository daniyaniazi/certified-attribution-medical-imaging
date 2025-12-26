"""
Evaluation runner: compute evaluation metrics on certified attributions.

Usage:
    python run_eval.py --checkpoint outputs/checkpoints/best_model.pt --method integrated_gradients
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
from src.certify.evaluate import CertificationEvaluator
from src.utils.io import load_checkpoint, load_attribution, save_json
from configs.defaults import DATASET_CONFIGS


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
    """Main evaluation script."""
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
    
    # Create evaluator
    evaluator = CertificationEvaluator()
    
    # Load attribution directory
    attr_dir = os.path.join(
        'outputs/attributions_raw',
        args.dataset,
        args.model,
        args.method
    )
    
    if not os.path.exists(attr_dir):
        print(f"Attribution directory not found: {attr_dir}")
        print("Run run_attribution.py first to generate attributions")
        return
    
    # Compute faithfulness for all samples
    print("\nComputing faithfulness (deletion AUC)...")
    faithfulness_scores = []
    
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Evaluating")
        
        for idx, batch in pbar:
            image = batch['image'].to(device)  # [1, C, H, W]
            label = batch['label'].item()
            meta = batch['meta']
            
            sample_id = meta['id'][0] if isinstance(meta['id'], list) else meta['id']
            
            # Load attribution
            attr_path = os.path.join(attr_dir, f'{sample_id}_attr.npy')
            if not os.path.exists(attr_path):
                continue
            
            attr = load_attribution(attr_path)
            
            # Normalize image for model input
            img_np = image[0].permute(1, 2, 0).cpu().numpy()
            
            # Compute faithfulness
            try:
                deletion_scores, auc = evaluator.compute_faithfulness_deletion(
                    model,
                    img_np,
                    attr,
                    target_class=1,
                    device=device,
                    num_steps=50
                )
                faithfulness_scores.append(auc)
                pbar.set_postfix({'auc': f'{auc:.3f}'})
            except Exception as e:
                print(f"Error computing faithfulness for {sample_id}: {e}")
                continue
    
    # Aggregate results
    if faithfulness_scores:
        eval_results = {
            'method': args.method,
            'model': args.model,
            'dataset': args.dataset,
            'num_samples': len(faithfulness_scores),
            'faithfulness': {
                'mean_auc': float(np.mean(faithfulness_scores)),
                'std_auc': float(np.std(faithfulness_scores)),
                'min_auc': float(np.min(faithfulness_scores)),
                'max_auc': float(np.max(faithfulness_scores))
            }
        }
        
        # Save evaluation report
        report_dir = 'outputs/reports'
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(
            report_dir,
            f'eval_{args.dataset}_{args.model}_{args.method}.json'
        )
        save_json(eval_results, report_path)
        
        print(f"\n=== Evaluation Results ===")
        print(f"Method: {args.method}")
        print(f"Model: {args.model}")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {len(faithfulness_scores)}")
        print(f"\nFaithfulness (Deletion AUC):")
        print(f"  Mean: {eval_results['faithfulness']['mean_auc']:.4f}")
        print(f"  Std:  {eval_results['faithfulness']['std_auc']:.4f}")
        print(f"  Min:  {eval_results['faithfulness']['min_auc']:.4f}")
        print(f"  Max:  {eval_results['faithfulness']['max_auc']:.4f}")
        
        print(f"\nReport saved to: {report_path}")
    else:
        print("No valid samples to evaluate")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate attribution maps')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='chexpert',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model name')
    parser.add_argument('--method', type=str, default='integrated_gradients',
                        choices=['integrated_gradients', 'gradcam', 'rise', 'occlusion'],
                        help='Attribution method')
    
    args = parser.parse_args()
    main(args)
