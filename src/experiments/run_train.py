"""
Training runner: train and save model checkpoints.

Usage:
    python run_train.py --dataset chexpert --model resnet18 --epochs 50
"""
import argparse
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.factory import get_model
from src.datasets.chexpert import CheXpertDataset
from src.datasets.isic import ISICDataset
from src.train.train_one import Trainer
from src.utils.seed import set_seed
from src.utils.io import save_json, save_config
from configs.defaults import TRAINING_CONFIG, DATASET_CONFIGS


def get_dataset_loader(dataset_name: str, split: str, config: dict):
    """Get dataset and DataLoader."""
    if dataset_name == 'chexpert':
        dataset = CheXpertDataset(
            root_dir=config.get('root_dir', 'data/raw/chexpert'),
            split=split,
            task=config.get('task', 'pneumonia'),
            transform=get_transform(config['target_size']),
            target_size=config['target_size']
        )
    elif dataset_name == 'isic':
        dataset = ISICDataset(
            root_dir=config.get('root_dir', 'data/raw/isic'),
            split=split,
            transform=get_transform(config['target_size']),
            target_size=config['target_size']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    loader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=(split == 'train'),
        num_workers=TRAINING_CONFIG['num_workers']
    )
    
    return dataset, loader


def get_transform(target_size: tuple):
    """Get preprocessing pipeline."""
    if target_size == 'chexpert':
        mean, std = (0.5026,), (0.2534,)
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def main(args):
    """Main training script."""
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join('outputs/checkpoints', args.dataset, args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get dataset config
    dataset_config = DATASET_CONFIGS.get(args.dataset)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"\n=== Training {args.model} on {args.dataset} ===")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, train_loader = get_dataset_loader(
        args.dataset, 'train', dataset_config
    )
    val_dataset, val_loader = get_dataset_loader(
        args.dataset, 'val', dataset_config
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Load model
    print(f"Loading model: {args.model}")
    model, config = get_model(
        args.model,
        num_classes=dataset_config['num_classes'],
        pretrained=True,
        device=device
    )
    
    total_params, trainable_params = __import__('src.models.factory', fromlist=['count_parameters']).count_parameters(model)
    print(f"Model params: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        task='binary' if dataset_config['num_classes'] == 2 else 'multi-class'
    )
    
    # Train
    history = trainer.fit(
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        metric_to_track=TRAINING_CONFIG['metric_to_track']
    )
    
    # Save history
    history_path = os.path.join(checkpoint_dir, 'history.json')
    save_json(history, history_path)
    print(f"\nTraining complete. Best checkpoint: {checkpoint_dir}/best_model.pt")
    print(f"History saved to: {history_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on medical imaging dataset')
    parser.add_argument('--dataset', type=str, default='chexpert',
                        choices=['chexpert', 'isic', 'aptos'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'efficientnet_b1'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    args = parser.parse_args()
    main(args)
