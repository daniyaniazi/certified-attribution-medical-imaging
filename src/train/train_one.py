"""Training loop with validation, checkpointing, and metrics."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from pathlib import Path

from src.train.metrics import MetricsComputer
from src.utils.io import save_checkpoint, load_checkpoint


class Trainer:
    """Training loop."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        task: str = 'binary'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: training dataloader
            val_loader: validation dataloader
            device: 'cpu' or 'cuda'
            learning_rate: learning rate
            weight_decay: L2 regularization
            task: 'binary' or 'multi-class'
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.task = task
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100  # Will be reset per epoch in train()
        )
        
        # Metrics
        self.metrics_computer = MetricsComputer()
        self.best_metric_value = -float('inf')
        self.best_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probas = []
        
        pbar = tqdm(self.train_loader, desc="Train", disable=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            probas = torch.softmax(logits, dim=1).detach().cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_probas.extend(probas)
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probas = np.array(all_probas)
        
        metrics = self.metrics_computer.compute_metrics(
            all_targets, all_preds, all_probas, task=self.task
        )
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on val set."""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probas = []
        
        pbar = tqdm(self.val_loader, desc="Val", disable=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(images)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()
            probas = torch.softmax(logits, dim=1).detach().cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            all_probas.extend(probas)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probas = np.array(all_probas)
        
        metrics = self.metrics_computer.compute_metrics(
            all_targets, all_preds, all_probas, task=self.task
        )
        
        return metrics
    
    def fit(
        self,
        epochs: int = 100,
        checkpoint_dir: str = 'outputs/checkpoints',
        metric_to_track: str = 'val_auc'
    ) -> Dict:
        """
        Train for multiple epochs.
        
        Args:
            epochs: number of epochs
            checkpoint_dir: where to save best checkpoint
            metric_to_track: which metric to use for best checkpoint
        
        Returns:
            history: dict of lists with train/val metrics per epoch
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_auc': [],
            'val_accuracy': [],
            'val_auc': [],
            'val_f1': []
        }
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Train: {self.metrics_computer.format_metrics(train_metrics)}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Val:   {self.metrics_computer.format_metrics(val_metrics)}")
            
            # Step scheduler
            self.scheduler.step()
            
            # Update history
            history['train_loss'].append(train_metrics.get('loss', 0))
            history['train_accuracy'].append(train_metrics.get('accuracy', 0))
            history['train_auc'].append(train_metrics.get('auc', 0))
            history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            history['val_auc'].append(val_metrics.get('auc', 0))
            history['val_f1'].append(val_metrics.get('f1', 0))
            
            # Save best checkpoint
            if metric_to_track in val_metrics:
                current_metric = val_metrics[metric_to_track]
                if current_metric > self.best_metric_value:
                    self.best_metric_value = current_metric
                    self.best_epoch = epoch
                    
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'best_model.pt'
                    )
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'metrics': val_metrics
                    }, checkpoint_path)
                    print(f"✓ Saved best checkpoint (epoch {epoch+1}, {metric_to_track}={current_metric:.4f}): {checkpoint_path}")
        
        return history
