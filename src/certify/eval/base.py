#!/usr/bin/env python
"""
Base evaluation class for certified attributions.

Provides common functionality:
- Loading models and datasets
- Saving/loading results
- Device management
- Directory structure management
"""

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
from tqdm import tqdm

import sys
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import get_model


class BaseEvaluator(ABC):
    """Abstract base class for certification evaluators."""
    
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        checkpoint_dir: Path,
        device: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            dataset_name: 'isic', 'chestxray', 'brain_mri', 'fundus'
            model_name: 'resnet18', 'densenet121', etc.
            checkpoint_dir: path to model checkpoints
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
    
    def _num_classes_for_dataset(self) -> int:
        """Return number of classes for the configured dataset."""
        name = self.dataset_name.lower()
        if name == 'isic':
            return 8
        # Default binary for others (chestxray, brain_mri, fundus, etc.)
        return 2

    def _load_model(self) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        model_dir = self.checkpoint_dir / self.model_name
        
        # Find latest checkpoint
        ckpts = sorted(model_dir.glob('*.pt')) + sorted(model_dir.glob('*.pth'))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")
        
        ckpt_path = ckpts[-1]
        print(f"  Loading checkpoint: {ckpt_path}")
        
        num_classes = self._num_classes_for_dataset()
        model_result = get_model(self.model_name, num_classes=num_classes, pretrained=False, device=self.device)
        # get_model may return (model, config); handle both
        if isinstance(model_result, tuple) and len(model_result) == 2:
            model, _ = model_result
        else:
            model = model_result
        state = torch.load(ckpt_path, map_location=self.device)

        # Unwrap trainer-style checkpoints
        if isinstance(state, dict):
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            elif 'state_dict' in state:
                state = state['state_dict']

        # Handle DataParallel wrapper
        if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
            state = {k.replace('module.', ''): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.to(self.device)
        return model
    
    def load_cert_results(self, cert_pkl: Path) -> Dict:
        """Load certification results pickle."""
        with open(cert_pkl, 'rb') as f:
            results = pickle.load(f)
        return results
    
    def save_results_json(self, results: Dict, output_path: Path) -> None:
        """Save results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved results to {output_path}")
    
    def save_results_pkl(self, results: Dict, output_path: Path) -> None:
        """Save results to pickle."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"  ✓ Saved results to {output_path}")
    
    @abstractmethod
    def evaluate_batch(
        self,
        cert_results_pkl: Path,
        dataset,
        output_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run evaluation on batch of certified results.
        
        Args:
            cert_results_pkl: path to certification results pickle
            dataset: PyTorch dataset
            output_dir: where to save results
            **kwargs: additional args for specific evaluator
        
        Returns:
            results dict
        """
        pass
    
    @abstractmethod
    def plot_results(self, results: Dict, output_dir: Path) -> None:
        """
        Generate plots from evaluation results.
        
        Args:
            results: evaluation results dict
            output_dir: where to save figures
        """
        pass
    
    def get_result_paths(self, output_base: Path) -> Dict[str, Path]:
        """
        Get standard result paths for dataset/model.
        
        Returns:
            {name -> path}
        """
        base_dir = output_base / self.dataset_name / self.model_name
        
        return {
            'results_json': base_dir / f'{self.__class__.__name__.lower()}_results.json',
            'results_pkl': base_dir / f'{self.__class__.__name__.lower()}_results.pkl',
            'figures_dir': base_dir / 'figures',
            'data_dir': base_dir / 'data',
        }
