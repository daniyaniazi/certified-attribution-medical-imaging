#!/usr/bin/env python
"""
Brain MRI Multi-Model Training Script for Server Execution

Trains multiple model architectures on the Brain MRI tumor dataset and generates
comprehensive evaluation reports and visualizations.

Usage:
    python train_brain_mri_server.py --epochs 50 --batch_size 32
    python train_brain_mri_server.py --models resnet18 resnet50 --epochs 100
    python train_brain_mri_server.py --no-eval  # Skip evaluation after training
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Add repo root to path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train.train_one import Trainer
from src.models.factory import get_model
from src.datasets.brain_mri import BrainMRIDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Brain MRI models on server')
    
    # Model configuration
    parser.add_argument('--models', nargs='+', 
                       default=['resnet18', 'resnet50', 'densenet121', 
                               'efficientnet_b0', 'efficientnet_b1', 'mobilenet_v2'],
                       help='Models to train (space-separated)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size (default: 224)')
    
    # Data paths
    parser.add_argument('--data_root', type=str, default='data/raw/brain_mri',
                       help='Path to Brain MRI dataset (default: data/raw/brain_mri)')
    
    # Output paths
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints/brain_mri',
                       help='Directory to save checkpoints (default: outputs/checkpoints/brain_mri)')
    parser.add_argument('--log_dir', type=str, default='outputs/logs/brain_mri',
                       help='Directory to save logs (default: outputs/logs/brain_mri)')
    parser.add_argument('--metrics_dir', type=str, default='outputs/metrics/brain_mri',
                       help='Directory to save metrics (default: outputs/metrics/brain_mri)')
    parser.add_argument('--reports_dir', type=str, default='outputs/reports',
                       help='Directory to save reports (default: outputs/reports)')
    
    # Training options
    parser.add_argument('--metric', type=str, default='val_auc',
                       choices=['val_auc', 'val_accuracy', 'val_f1'],
                       help='Metric to track for best model (default: val_auc)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='Do not use ImageNet pretrained weights')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing checkpoints if available')
    
    # Evaluation options
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only run evaluation on existing checkpoints')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: cuda, cpu, or auto (default: auto)')
    
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
    log_root = Path(args.log_dir)
    metrics_root = Path(args.metrics_dir)
    reports_root = Path(args.reports_dir)
    
    for p in (checkpoint_root, log_root, metrics_root, reports_root):
        p.mkdir(parents=True, exist_ok=True)
    
    print(f'\nOutput directories:')
    print(f'  Checkpoints: {checkpoint_root.absolute()}')
    print(f'  Logs: {log_root.absolute()}')
    print(f'  Metrics: {metrics_root.absolute()}')
    print(f'  Reports: {reports_root.absolute()}')
    
    return device, checkpoint_root, log_root, metrics_root, reports_root


def verify_data(data_root):
    """Verify data directory structure."""
    print('\n' + '='*80)
    print('DATA VERIFICATION')
    print('='*80)
    
    data_path = ROOT / data_root
    print(f'Data root: {data_path.absolute()}')
    
    if not data_path.exists():
        raise FileNotFoundError(f'Data directory not found: {data_path.absolute()}')
    
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        images_dir = split_dir / 'images'
        labels_file = split_dir / 'labels.json'
        
        print(f'\n{split}/:')
        print(f'  Directory: {split_dir.exists()}')
        print(f'  images/: {images_dir.exists()}')
        print(f'  labels.json: {labels_file.exists()}')
        
        if images_dir.exists():
            image_files = [f for f in os.listdir(images_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            print(f'  Image count: {len(image_files)}')
    
    print('\n✓ Data verification complete')
    return data_path


def load_datasets(data_root, img_size, batch_size, num_workers):
    """Load and prepare datasets."""
    print('\n' + '='*80)
    print('LOADING DATASETS')
    print('='*80)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print(f'Loading datasets with image size {img_size}x{img_size}...')
    train_dataset = BrainMRIDataset(
        str(data_root), 
        split='train', 
        transform=train_transform, 
        target_size=(img_size, img_size)
    )
    val_dataset = BrainMRIDataset(
        str(data_root), 
        split='val', 
        transform=val_transform, 
        target_size=(img_size, img_size)
    )
    
    print(f'✓ Train dataset: {len(train_dataset)} samples')
    print(f'✓ Val dataset: {len(val_dataset)} samples')
    
    # Infer number of classes
    num_classes = infer_num_classes(train_dataset)
    print(f'✓ Number of classes: {num_classes}')
    
    # Create data loaders
    print(f'\nCreating DataLoaders (batch_size={batch_size}, workers={num_workers})...')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f'✓ Train loader: {len(train_loader)} batches')
    print(f'✓ Val loader: {len(val_loader)} batches')
    
    return train_loader, val_loader, num_classes, val_dataset


def infer_num_classes(dataset):
    """Infer number of classes from dataset."""
    for attr in ('num_classes', 'n_classes', 'classes', 'label_map'):
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            if isinstance(val, (list, tuple, dict)):
                return len(val)
            if isinstance(val, int):
                return val
    
    # Fallback: scan dataset
    try:
        labels = [dataset[i]['label'] for i in range(min(len(dataset), 200))]
        return len(set(int(l) for l in labels))
    except Exception:
        return 4  # Brain MRI default (normal, glioma, meningioma, pituitary)


def train_model(model_name, train_loader, val_loader, num_classes, 
                device, args, checkpoint_dir, log_dir, metrics_dir):
    """Train a single model."""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")
    
    # Create model-specific directories
    ckpt_dir = checkpoint_dir / model_name
    log_dir_model = log_dir / model_name
    metrics_dir_model = metrics_dir / model_name
    
    for p in (ckpt_dir, log_dir_model, metrics_dir_model):
        p.mkdir(parents=True, exist_ok=True)
    
    # Build model
    model, cfg = get_model(
        model_name, 
        num_classes=num_classes, 
        pretrained=not args.no_pretrained, 
        device=device
    )
    
    # Check for existing checkpoint
    best_ckpt = ckpt_dir / 'best_model.pt'
    final_ckpt = ckpt_dir / 'final_model.pt'
    start_epoch = 0
    existing_history = {}
    
    if args.resume:
        if best_ckpt.exists():
            print(f'📂 Loading checkpoint: {best_ckpt}')
            checkpoint = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f'   ✓ Resuming from epoch {start_epoch}')
        elif final_ckpt.exists():
            print(f'📂 Loading checkpoint: {final_ckpt}')
            checkpoint = torch.load(final_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f'   ✓ Resuming from epoch {start_epoch}')
        else:
            print(f'✓ Model initialized: {cfg.backbone} (pretrained={not args.no_pretrained})')
    else:
        print(f'✓ Model initialized: {cfg.backbone} (pretrained={not args.no_pretrained})')
    
    # Load existing history
    hist_path = log_dir_model / 'history.json'
    if hist_path.exists() and args.resume:
        print(f'📂 Loading history from {hist_path}')
        with open(hist_path, 'r') as f:
            existing_history = json.load(f)
        print(f'   ✓ Found {len(existing_history.get("train_loss", []))} previous epochs')
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        task='multi-class' if num_classes > 2 else 'binary'
    )
    print(f'✓ Trainer initialized')
    
    # Train
    print(f'\nTraining for {args.epochs} epochs...')
    new_history = trainer.fit(
        epochs=args.epochs,
        checkpoint_dir=str(ckpt_dir),
        metric_to_track=args.metric
    )
    
    # Merge histories
    if existing_history:
        for key in new_history:
            if key in existing_history:
                existing_history[key].extend(new_history[key])
            else:
                existing_history[key] = new_history[key]
        combined_history = existing_history
        print(f'✓ Merged history ({len(combined_history.get("train_loss", []))} total epochs)')
    else:
        combined_history = new_history
    
    # Save history
    with open(hist_path, 'w') as f:
        json.dump(combined_history, f, indent=2)
    print(f'✓ History saved: {hist_path}')
    
    # Save final model
    final_model_path = ckpt_dir / 'final_model.pt'
    torch.save({
        'epoch': start_epoch + args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'metrics': {
            'final_train_loss': combined_history.get('train_loss', [None])[-1],
            'final_val_auc': combined_history.get('val_auc', [None])[-1],
            'final_val_accuracy': combined_history.get('val_accuracy', [None])[-1]
        }
    }, final_model_path)
    print(f'✓ Final model saved: {final_model_path}')
    
    # Save metrics
    checkpoint_to_use = best_ckpt if best_ckpt.exists() else final_model_path
    metrics_out = {}
    
    try:
        data = torch.load(checkpoint_to_use, map_location='cpu')
        if isinstance(data, dict) and 'metrics' in data:
            metrics_out = data['metrics']
    except Exception as e:
        print(f'⚠ Could not load metrics: {e}')
    
    # Add history summary
    try:
        hist_summary = {
            k: (v[-1] if isinstance(v, list) and len(v) > 0 else None)
            for k, v in combined_history.items()
        }
        metrics_out['history_summary'] = hist_summary
        metrics_out['total_epochs'] = len(combined_history.get('train_loss', []))
    except Exception:
        pass
    
    metrics_path = metrics_dir_model / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'✓ Metrics saved: {metrics_path}')
    
    return {
        'checkpoint': str(checkpoint_to_use),
        'history': str(hist_path),
        'metrics': str(metrics_path),
        'total_epochs': len(combined_history.get('train_loss', []))
    }


def evaluate_models(models, val_loader, num_classes, device, 
                    checkpoint_dir, log_dir, reports_dir, val_dataset):
    """Evaluate all trained models."""
    print('\n' + '='*80)
    print('EVALUATING ALL MODELS')
    print('='*80)
    
    eval_results = {}
    
    for model_name in models:
        print(f'\n{model_name}:')
        
        # Load checkpoint
        ckpt_path = checkpoint_dir / model_name / 'best_model.pt'
        if not ckpt_path.exists():
            ckpt_path = checkpoint_dir / model_name / 'final_model.pt'
        
        if not ckpt_path.exists():
            print(f'  ⚠ No checkpoint found, skipping...')
            continue
        
        # Load model
        model, cfg = get_model(model_name, num_classes=num_classes, 
                              pretrained=False, device=device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f'  ✓ Model loaded from {ckpt_path.name}')
        
        # Evaluate
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'  Evaluating', leave=False):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        print(f'  ✓ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
        
        eval_results[model_name] = {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'checkpoint_metrics': checkpoint.get('metrics', {})
        }
    
    if not eval_results:
        print('\n⚠ No models to evaluate')
        return None, None, None
    
    # Find best model
    sorted_models = sorted(eval_results.items(), 
                          key=lambda x: x[1]['accuracy'], reverse=True)
    best_model_name = sorted_models[0][0]
    best_accuracy = sorted_models[0][1]['accuracy']
    
    print('\n' + '='*80)
    print('RANKING')
    print('='*80)
    print(f"\n{'Rank':<6} {'Model':<20} {'Accuracy':<12}")
    print('-' * 60)
    for i, (model_name, result) in enumerate(sorted_models, 1):
        acc = result['accuracy']
        print(f"{i:<6} {model_name:<20} {acc:.4f} ({acc*100:.2f}%)")
    
    print(f'\n🏆 BEST MODEL: {best_model_name} ({best_accuracy*100:.2f}%)')
    
    # Generate visualizations
    generate_visualizations(models, eval_results, best_model_name, 
                           log_dir, reports_dir, val_dataset, num_classes)
    
    return eval_results, best_model_name, best_accuracy


def generate_visualizations(models, eval_results, best_model_name, 
                            log_dir, reports_dir, val_dataset, num_classes):
    """Generate all visualization plots."""
    print('\n' + '='*80)
    print('GENERATING VISUALIZATIONS')
    print('='*80)
    
    # Get class names
    try:
        if hasattr(val_dataset, 'classes'):
            class_names = val_dataset.classes
        elif hasattr(val_dataset, 'label_map'):
            class_names = list(val_dataset.label_map.values())
        else:
            class_names = [f'Class {i}' for i in range(num_classes)]
    except:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # 1. Training curves
    plot_training_curves(models, log_dir, reports_dir)
    
    # 2. Model comparison
    plot_model_comparison(eval_results, best_model_name, reports_dir)
    
    # 3. Confusion matrix
    plot_confusion_matrix(eval_results[best_model_name], best_model_name, 
                         class_names, reports_dir)
    
    # 4. Classification report
    save_classification_report(eval_results[best_model_name], best_model_name, 
                              class_names, reports_dir)
    
    print('✓ All visualizations saved')


def plot_training_curves(models, log_dir, reports_dir):
    """Plot training curves for all models."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History - All Models', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models[:6]):
        hist_path = log_dir / model_name / 'history.json'
        
        if not hist_path.exists():
            axes[idx].text(0.5, 0.5, f'{model_name}\nNo history', 
                          ha='center', va='center')
            axes[idx].set_title(model_name)
            continue
        
        with open(hist_path, 'r') as f:
            history = json.load(f)
        
        ax = axes[idx]
        epochs_range = range(1, len(history.get('train_loss', [])) + 1)
        
        if 'train_loss' in history and history['train_loss']:
            ax.plot(epochs_range, history['train_loss'], 'b-', 
                   label='Train Loss', linewidth=2)
        
        ax2 = ax.twinx()
        if 'val_accuracy' in history and history['val_accuracy']:
            ax2.plot(epochs_range, history['val_accuracy'], 'r-', 
                    label='Val Accuracy', linewidth=2)
        if 'val_auc' in history and history['val_auc']:
            ax2.plot(epochs_range, history['val_auc'], 'g--', 
                    label='Val AUC', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', color='b', fontsize=10)
        ax2.set_ylabel('Accuracy / AUC', color='r', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    save_path = reports_dir / 'brain_mri_training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Training curves: {save_path}')


def plot_model_comparison(eval_results, best_model_name, reports_dir):
    """Plot model comparison bar chart."""
    plt.figure(figsize=(12, 6))
    
    model_names = list(eval_results.keys())
    accuracies = [eval_results[m]['accuracy'] * 100 for m in model_names]
    
    colors = ['gold' if m == best_model_name else 'steelblue' for m in model_names]
    bars = plt.bar(range(len(model_names)), accuracies, color=colors, 
                   edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Model Comparison - Validation Accuracy (Brain MRI)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    save_path = reports_dir / 'brain_mri_model_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Model comparison: {save_path}')


def plot_confusion_matrix(result, model_name, class_names, reports_dir):
    """Plot confusion matrix for best model."""
    cm = confusion_matrix(result['labels'], result['predictions'])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name} (Brain MRI)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = reports_dir / 'brain_mri_confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓ Confusion matrix: {save_path}')


def save_classification_report(result, model_name, class_names, reports_dir):
    """Save classification report."""
    report = classification_report(result['labels'], result['predictions'],
                                   target_names=class_names, digits=4)
    
    report_path = reports_dir / 'brain_mri_classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {model_name} (Brain MRI)\n")
        f.write("="*80 + "\n\n")
        f.write(report)
    
    print(f'  ✓ Classification report: {report_path}')


def main():
    """Main execution function."""
    args = parse_args()
    
    print('\n' + '='*80)
    print('BRAIN MRI MULTI-MODEL TRAINING')
    print('='*80)
    print(f'Models: {", ".join(args.models)}')
    print(f'Epochs: {args.epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'Image size: {args.img_size}')
    print(f'Metric to track: {args.metric}')
    print(f'Resume: {args.resume}')
    print(f'Eval only: {args.eval_only}')
    
    # Setup
    device, checkpoint_dir, log_dir, metrics_dir, reports_dir = setup_environment(args)
    
    # Verify data
    data_root = verify_data(args.data_root)
    
    # Load datasets
    train_loader, val_loader, num_classes, val_dataset = load_datasets(
        data_root, args.img_size, args.batch_size, args.num_workers
    )
    
    # Training
    results = {}
    if not args.eval_only:
        print('\n' + '='*80)
        print('STARTING TRAINING')
        print('='*80)
        
        for model_name in args.models:
            result = train_model(
                model_name, train_loader, val_loader, num_classes,
                device, args, checkpoint_dir, log_dir, metrics_dir
            )
            results[model_name] = result
        
        print('\n' + '='*80)
        print('TRAINING COMPLETE')
        print('='*80)
        for model_name, info in results.items():
            print(f'\n{model_name}:')
            for key, val in info.items():
                print(f'  {key}: {val}')
    
    # Evaluation
    if not args.no_eval:
        eval_results, best_model, best_acc = evaluate_models(
            args.models, val_loader, num_classes, device,
            checkpoint_dir, log_dir, reports_dir, val_dataset
        )
        
        if eval_results:
            print('\n' + '='*80)
            print('FINAL SUMMARY')
            print('='*80)
            print(f'🏆 Best Model: {best_model}')
            print(f'   Accuracy: {best_acc*100:.2f}%')
            print(f'\nAll outputs saved to:')
            print(f'  - Checkpoints: {checkpoint_dir}')
            print(f'  - Logs: {log_dir}')
            print(f'  - Metrics: {metrics_dir}')
            print(f'  - Reports: {reports_dir}')
    
    print('\n' + '='*80)
    print('DONE')
    print('='*80)


if __name__ == '__main__':
    main()
