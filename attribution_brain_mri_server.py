#!/usr/bin/env python
"""
Brain MRI Attribution Generation Script for Server Execution

Generates attributions (saliency maps) for trained Brain MRI models using multiple
attribution methods and saves results for visualization and analysis.

Usage:
    python attribution_brain_mri_server.py --num_samples 10
    python attribution_brain_mri_server.py --models resnet18 resnet50 --num_samples 20
    python attribution_brain_mri_server.py --methods IntegratedGradients GradCAM --no-viz
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.ndimage import zoom

# Add repo root to path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import get_model
from src.datasets.brain_mri import BrainMRIDataset
from src.xai.attribution_unified import (
    IntegratedGradientsUnified,
    GradCAMUnified,
    RISEUnified,
    OcclusionUnified,
    LRPUnified
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate attributions for Brain MRI models on server')
    
    # Model configuration
    parser.add_argument('--models', nargs='+',
                       default=['resnet18', 'resnet50', 'densenet121',
                               'efficientnet_b0', 'efficientnet_b1', 'mobilenet_v2'],
                       help='Models to process (space-separated)')
    
    # Attribution methods
    parser.add_argument('--methods', nargs='+',
                       default=['IntegratedGradients', 'GradCAM', 'RISE', 'Occlusion', 'LRP'],
                       choices=['IntegratedGradients', 'GradCAM', 'RISE', 'Occlusion', 'LRP'],
                       help='Attribution methods to use (space-separated)')
    
    # Data configuration
    parser.add_argument('--data_root', type=str, default='data/raw/brain_mri',
                       help='Path to Brain MRI dataset (default: data/raw/brain_mri)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (default: 224)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to process (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    
    # Model paths
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints/brain_mri',
                       help='Directory with trained model checkpoints (default: outputs/checkpoints/brain_mri)')
    
    # Output paths
    parser.add_argument('--output_dir', type=str, default='outputs/attributions_viz/brain_mri',
                       help='Directory to save attributions (default: outputs/attributions_viz/brain_mri)')
    parser.add_argument('--log_dir', type=str, default='outputs/logs/attribution_brain_mri',
                       help='Directory to save logs (default: outputs/logs/attribution_brain_mri)')
    
    # Attribution parameters
    parser.add_argument('--rise_samples', type=int, default=500,
                       help='Number of samples for RISE (default: 500)')
    parser.add_argument('--ig_steps', type=int, default=50,
                       help='Number of steps for Integrated Gradients (default: 50)')
    parser.add_argument('--occlusion_patch', type=int, default=8,
                       help='Patch size for Occlusion (default: 8)')
    parser.add_argument('--lrp_epsilon', type=float, default=1e-6,
                       help='Epsilon for LRP (default: 1e-6)')
    
    # Visualization options
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing results')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: cuda, cpu, or auto (default: auto)')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup paths and device."""
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print('='*80)
    print('ENVIRONMENT SETUP')
    print('='*80)
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA Version: {torch.version.cuda}')
    print(f'PyTorch Version: {torch.__version__}')
    print(f'Repo Root: {ROOT}')
    
    # Create output directories
    checkpoint_root = Path(args.checkpoint_dir)
    output_root = Path(args.output_dir)
    log_root = Path(args.log_dir)
    
    for p in (checkpoint_root, output_root, log_root):
        p.mkdir(parents=True, exist_ok=True)
    
    print(f'\nOutput directories:')
    print(f'  Checkpoints: {checkpoint_root.absolute()}')
    print(f'  Attributions: {output_root.absolute()}')
    print(f'  Logs: {log_root.absolute()}')
    print('='*80)
    
    return device


def discover_models(checkpoint_dir):
    """Discover trained models in checkpoint directory."""
    print('\nDiscovering trained models...')
    
    checkpoint_path = Path(checkpoint_dir)
    available_models = []
    
    if not checkpoint_path.exists():
        print(f'⚠ Checkpoint directory not found: {checkpoint_path.absolute()}')
        return available_models
    
    for model_dir in sorted(checkpoint_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        best_ckpt = model_dir / 'best_model.pt'
        final_ckpt = model_dir / 'final_model.pt'
        
        if best_ckpt.exists() or final_ckpt.exists():
            checkpoint_file = best_ckpt if best_ckpt.exists() else final_ckpt
            available_models.append({
                'name': model_name,
                'checkpoint': checkpoint_file,
                'type': 'best' if best_ckpt.exists() else 'final'
            })
    
    print(f'Found {len(available_models)} trained models:')
    for i, m in enumerate(available_models, 1):
        print(f'  {i}. {m["name"]:<20} ({m["type"]} checkpoint)')
    
    return available_models


def load_dataset(data_root, img_size, num_samples):
    """Load Brain MRI validation dataset."""
    print('\nLoading Brain MRI validation dataset...')
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    data_path = ROOT / data_root if not Path(data_root).is_absolute() else Path(data_root)
    
    val_dataset = BrainMRIDataset(
        str(data_path),
        split='val',
        transform=val_transform,
        target_size=(img_size, img_size)
    )
    
    print(f'✓ Loaded {len(val_dataset)} validation images')
    
    # Get class names
    if hasattr(val_dataset, 'classes'):
        class_names = val_dataset.classes
    elif hasattr(val_dataset, 'label_map'):
        class_names = list(val_dataset.label_map.values())
    else:
        class_names = ['normal', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
    
    print(f'Classes: {class_names}')
    
    return val_dataset, class_names


def get_target_layer(model, model_name):
    """Get last convolutional layer for Grad-CAM."""
    if 'resnet' in model_name.lower():
        return model.layer4[-1].conv2 if hasattr(model.layer4[-1], 'conv2') else model.layer4[-1]
    elif 'densenet' in model_name.lower():
        return model.features.denseblock4
    elif 'efficientnet' in model_name.lower():
        return list(model.features.modules())[-1]
    elif 'mobilenet' in model_name.lower():
        return list(model.features.modules())[-1]
    else:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError(f'Could not find target layer for {model_name}')


def normalize_image(image):
    """Apply ImageNet normalization."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return normalize(image)


def to_numpy_2d(arr):
    """Convert array to 2D numpy."""
    arr_np = np.array(arr)
    while arr_np.ndim > 2:
        if arr_np.shape[0] == 1:
            arr_np = arr_np.squeeze(0)
        elif arr_np.shape[-1] == 1:
            arr_np = arr_np.squeeze(-1)
        else:
            if arr_np.ndim == 3:
                arr_np = arr_np.mean(axis=0)
            elif arr_np.ndim == 4:
                arr_np = arr_np.mean(axis=(0, 1))
            break
    if arr_np.ndim != 2:
        raise ValueError(f'Cannot reduce to 2D, shape: {arr_np.shape}')
    return arr_np.astype(np.float32)


def generate_attributions(model_name, model, attribution_methods, val_dataset,
                         class_names, sample_indices, args, device):
    """Generate attributions for all samples and methods."""
    print(f'\n  Processing {len(sample_indices)} samples...')
    
    results = {
        'predictions': [],
        'attributions': {method: [] for method in attribution_methods.keys()}
    }
    
    for sample_idx in tqdm(sample_indices, desc=f'    Samples', leave=False):
        sample = val_dataset[sample_idx]
        image = sample['image']
        true_label = sample['label']
        
        # Get prediction
        with torch.no_grad():
            normalized_image = normalize_image(image).unsqueeze(0).to(device)
            output = model(normalized_image)
            pred_label = output.argmax(dim=1).item()
            pred_prob = torch.softmax(output, dim=1)[0, pred_label].item()
        
        results['predictions'].append({
            'sample_idx': int(sample_idx),
            'true_label': int(true_label),
            'pred_label': int(pred_label),
            'pred_prob': float(pred_prob),
            'correct': int(true_label) == int(pred_label)
        })
        
        # Generate attributions
        normalized_image_single = normalize_image(image)
        
        for method_name, method_obj in attribution_methods.items():
            try:
                # Build kwargs based on method
                kwargs = {}
                if method_name == 'IntegratedGradients':
                    kwargs['num_steps'] = args.ig_steps
                elif method_name == 'RISE':
                    kwargs['num_samples'] = args.rise_samples
                    kwargs['mask_size'] = 14
                elif method_name == 'Occlusion':
                    kwargs['patch_size'] = args.occlusion_patch
                    kwargs['stride'] = max(1, args.occlusion_patch // 2)
                
                heatmap = method_obj.attribute(
                    normalized_image_single,
                    target_class=pred_label,
                    **kwargs
                )
                
                heatmap_np = to_numpy_2d(heatmap)
                results['attributions'][method_name].append(heatmap_np)
                
            except Exception as e:
                print(f'      ✗ {method_name} failed: {e}')
                results['attributions'][method_name].append(None)
    
    return results


def process_models(available_models, args, device, val_dataset, class_names):
    """Process all available models."""
    print('\n' + '='*80)
    print('ATTRIBUTION GENERATION')
    print('='*80)
    
    # Select models to process
    models_to_process = [m for m in available_models if m['name'] in args.models]
    if not models_to_process:
        print('⚠ No matching models found to process')
        return {}
    
    all_results = {}
    output_root = Path(args.output_dir)
    
    # Prepare sample indices
    np.random.seed(42)
    sample_indices = np.random.choice(len(val_dataset), 
                                     min(args.num_samples, len(val_dataset)),
                                     replace=False)
    
    for model_info in tqdm(models_to_process, desc='Models'):
        model_name = model_info['name']
        checkpoint_path = model_info['checkpoint']
        
        print(f'\n{"="*80}')
        print(f'Processing: {model_name}')
        print(f'{"="*80}')
        
        # Load model
        try:
            model, cfg = get_model(model_name, num_classes=4, 
                                 pretrained=False, device=device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f'✓ Model loaded')
        except Exception as e:
            print(f'✗ Failed to load model: {e}')
            continue
        
        # Initialize attribution methods
        try:
            target_layer = get_target_layer(model, model_name)
            
            attribution_methods = {
                'IntegratedGradients': IntegratedGradientsUnified(model, device),
                'GradCAM': GradCAMUnified(model, target_layer, device),
                'RISE': RISEUnified(model, device),
                'Occlusion': OcclusionUnified(model, device),
                'LRP': LRPUnified(model, device, epsilon=args.lrp_epsilon)
            }
            
            # Filter to requested methods
            attribution_methods = {k: v for k, v in attribution_methods.items()
                                 if k in args.methods}
            
            print(f'✓ Initialized {len(attribution_methods)} attribution methods')
        except Exception as e:
            print(f'✗ Failed to initialize methods: {e}')
            continue
        
        # Generate attributions
        try:
            results = generate_attributions(model_name, model, attribution_methods,
                                          val_dataset, class_names, sample_indices,
                                          args, device)
            all_results[model_name] = results
            print(f'✓ Attributions generated')
        except Exception as e:
            print(f'✗ Failed to generate attributions: {e}')
            continue
        
        # Save heatmaps for this model
        try:
            model_output_dir = output_root / model_name
            
            for method_name, heatmaps in results['attributions'].items():
                method_dir = model_output_dir / method_name
                method_dir.mkdir(parents=True, exist_ok=True)
                
                for i, heatmap in enumerate(heatmaps):
                    if heatmap is not None:
                        sample_idx = sample_indices[i]
                        heatmap_file = method_dir / f'sample_{sample_idx}.npy'
                        np.save(heatmap_file, heatmap)
            
            print(f'✓ Heatmaps saved')
        except Exception as e:
            print(f'✗ Failed to save heatmaps: {e}')
    
    return all_results


def save_results(all_results, args, class_names):
    """Save results to disk."""
    print('\n' + '='*80)
    print('SAVING RESULTS')
    print('='*80)
    
    output_root = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save predictions as JSON
    predictions_file = output_root / f'predictions_{timestamp}.json'
    predictions_data = {}
    
    for model_name, results in all_results.items():
        predictions_data[model_name] = results['predictions']
    
    try:
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f'✓ Saved predictions: {predictions_file.name}')
    except Exception as e:
        print(f'✗ Failed to save predictions: {e}')
    
    # Save index
    index_file = output_root / 'index.json'
    try:
        available_models = list(all_results.keys())
        available_methods = list(next(iter(all_results.values()))['attributions'].keys()) \
                          if all_results else []
        
        index = {
            'timestamp': timestamp,
            'num_models': len(all_results),
            'models': available_models,
            'num_samples': len(next(iter(all_results.values()))['predictions']) \
                          if all_results else 0,
            'methods': available_methods,
            'class_names': class_names,
            'heatmap_structure': 'model_name/method_name/sample_*.npy'
        }
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        print(f'✓ Saved index: {index_file.name}')
    except Exception as e:
        print(f'✗ Failed to save index: {e}')
    
    print('='*80)
    print(f'Results saved to: {output_root.absolute()}')
    print('='*80)


def overlay_heatmap(image, heatmap, alpha=0.5, colormap='jet'):
    """Overlay heatmap on image."""
    if heatmap.shape != image.shape[:2]:
        zoom_factor = (image.shape[0] / heatmap.shape[0], 
                      image.shape[1] / heatmap.shape[1])
        heatmap = zoom(heatmap, zoom_factor, order=1)
    
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]
    
    overlaid = (1 - alpha) * image + alpha * heatmap_colored
    return np.clip(overlaid, 0, 1)


def load_heatmap(model_name, method_name, sample_idx, output_root):
    """Load a single heatmap from disk."""
    heatmap_file = output_root / model_name / method_name / f'sample_{sample_idx}.npy'
    if heatmap_file.exists():
        return np.load(heatmap_file)
    return None


def visualize_results(all_results, args, val_dataset, class_names):
    """Generate visualization plots."""
    if args.no_viz or not all_results:
        return
    
    print('\n' + '='*80)
    print('GENERATING VISUALIZATIONS')
    print('='*80)
    
    output_root = Path(args.output_dir)
    
    # Get all sample indices and methods from first model's results
    first_results = next(iter(all_results.values()))
    sample_indices = np.array([p['sample_idx'] for p in first_results['predictions']])
    available_methods = list(first_results['attributions'].keys())
    available_models = list(all_results.keys())
    
    # 1. Visualize each sample across all models
    for idx, sample_idx in enumerate(sample_indices[:min(5, len(sample_indices))]):
        try:
            sample = val_dataset[sample_idx]
            image = sample['image'].permute(1, 2, 0).numpy()
            true_label = sample['label']
            true_class = class_names[true_label] if true_label < len(class_names) else f'Class {true_label}'
            
            num_models = len(available_models)
            num_cols = 1 + len(available_methods)
            
            fig, axes = plt.subplots(num_models, num_cols, figsize=(4 * num_cols, 4 * num_models))
            if num_models == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle(f'Sample {sample_idx} - True: {true_class}', 
                        fontsize=14, fontweight='bold')
            
            for row_idx, model_name in enumerate(available_models):
                # Original image
                axes[row_idx, 0].imshow(image)
                axes[row_idx, 0].axis('off')
                axes[row_idx, 0].set_title(f'{model_name}')
                
                # Attributions
                for col_idx, method_name in enumerate(available_methods, start=1):
                    heatmap = load_heatmap(model_name, method_name, sample_idx, output_root)
                    
                    if heatmap is not None:
                        overlaid = overlay_heatmap(image, heatmap, alpha=0.5)
                        axes[row_idx, col_idx].imshow(overlaid)
                    else:
                        axes[row_idx, col_idx].text(0.5, 0.5, 'N/A',
                                                   ha='center', va='center',
                                                   transform=axes[row_idx, col_idx].transAxes)
                    
                    axes[row_idx, col_idx].axis('off')
                    if row_idx == 0:
                        axes[row_idx, col_idx].set_title(method_name)
            
            plt.tight_layout()
            save_path = output_root / f'sample_{sample_idx}_all_models.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'  ✓ Saved: sample_{sample_idx}_all_models.png')
            
        except Exception as e:
            print(f'  ✗ Failed to visualize sample {sample_idx}: {e}')
    
    print('✓ Visualizations complete')


def main():
    """Main execution."""
    args = parse_args()
    
    # Setup
    device = setup_environment(args)
    
    # Discover and load
    available_models = discover_models(args.checkpoint_dir)
    if not available_models:
        print('No models found. Exiting.')
        return
    
    val_dataset, class_names = load_dataset(args.data_root, args.img_size, args.num_samples)
    
    # Process
    all_results = process_models(available_models, args, device, val_dataset, class_names)
    
    if not all_results:
        print('No results generated. Exiting.')
        return
    
    # Save and visualize
    save_results(all_results, args, class_names)
    visualize_results(all_results, args, val_dataset, class_names)
    
    print('\n' + '='*80)
    print('✅ ATTRIBUTION GENERATION COMPLETE')
    print('='*80)
    print(f'Models processed: {len(all_results)}')
    print(f'Samples per model: {len(next(iter(all_results.values()))["predictions"])}')
    print(f'Output directory: {Path(args.output_dir).absolute()}')
    print('='*80)


if __name__ == '__main__':
    main()
