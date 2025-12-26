"""
Unified attribution interface matching paper specification.

All methods return heatmaps [H,W] normalized to [0,1].
This matches the paper's framework where h(x) → heatmap.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class AttributionMethod(ABC):
    """Base class for all attribution methods."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.eval()
    
    @abstractmethod
    def attribute(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Compute attribution heatmap.
        
        Args:
            image: input image [C,H,W] or [1,C,H,W]
            target_class: target class index
        
        Returns:
            heatmap [H,W] normalized to [0,1]
        """
        pass
    
    def _normalize_attribution(self, attr: np.ndarray) -> np.ndarray:
        """Normalize attribution to [0,1]."""
        attr_min = attr.min()
        attr_max = attr.max()
        if attr_max > attr_min:
            return (attr - attr_min) / (attr_max - attr_min + 1e-10)
        else:
            return attr.astype(np.float32)


class IntegratedGradientsUnified(AttributionMethod):
    """Integrated Gradients - unified interface."""
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int,
        num_steps: int = 50,
        baseline: torch.Tensor = None
    ) -> np.ndarray:
        """Integrated Gradients attribution."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        if baseline is None:
            baseline = torch.zeros_like(image)
        
        baseline = baseline.to(self.device)
        image = image.to(self.device)
        
        alphas = torch.linspace(0, 1, num_steps + 1, device=self.device)
        accumulated_grads = None
        
        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad_(True)
            
            output = self.model(interpolated)
            logit = output[0, target_class]
            
            if interpolated.grad is not None:
                interpolated.grad.zero_()
            
            grads = torch.autograd.grad(
                logit,
                interpolated,
                create_graph=False,
                only_inputs=True
            )[0]
            
            if accumulated_grads is None:
                accumulated_grads = grads.clone()
            else:
                accumulated_grads += grads
        
        avg_grads = accumulated_grads / len(alphas)
        integrated_grads = (image - baseline) * avg_grads
        
        attribution = torch.sum(integrated_grads, dim=1)[0]
        attribution = attribution.abs().detach().cpu().numpy()
        
        return self._normalize_attribution(attribution)


class GradCAMUnified(AttributionMethod):
    """Grad-CAM - unified interface."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = 'cpu'):
        super().__init__(model, device)
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self._save_features)
        self.target_layer.register_backward_hook(self._save_gradients)
    
    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def attribute(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """Grad-CAM attribution."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad_(True)
        
        output = self.model(image)
        logit = output[0, target_class]
        
        self.model.zero_grad()
        logit.backward(retain_graph=False)
        
        gradients = self.gradients
        feature_maps = self.feature_maps
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1)[0]
        
        import torch.nn.functional as F
        cam = F.relu(cam)
        
        cam_np = cam.detach().cpu().numpy()
        return self._normalize_attribution(cam_np)


class RISEUnified(AttributionMethod):
    """RISE - unified interface."""
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int,
        num_samples: int = 1000,
        mask_size: int = 7,
        prob_include: float = 0.5
    ) -> np.ndarray:
        """RISE attribution."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        _, c, h, w = image.shape
        image = image.to(self.device)
        
        attribution = np.zeros((h, w))
        
        import torch.nn.functional as F
        from tqdm import tqdm
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="RISE", disable=True):
                mask_shape = (1, 1, mask_size, mask_size)
                mask = (torch.rand(mask_shape) < prob_include).float()
                
                mask = F.interpolate(mask, size=(h, w), mode='nearest')
                masked_image = image * mask
                
                output = self.model(masked_image)
                prob = torch.softmax(output, dim=1)[0, target_class].item()
                
                attribution += prob * mask[0, 0].cpu().numpy()
        
        attribution = attribution / num_samples
        return self._normalize_attribution(attribution)


class OcclusionUnified(AttributionMethod):
    """Occlusion - unified interface."""
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int,
        patch_size: int = 16,
        stride: int = 8
    ) -> np.ndarray:
        """Occlusion attribution."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        _, c, h, w = image.shape
        image = image.to(self.device)
        
        with torch.no_grad():
            baseline_output = self.model(image)
            baseline_prob = torch.softmax(baseline_output, dim=1)[0, target_class].item()
        
        attribution = np.zeros((h, w))
        count = np.zeros((h, w))
        
        with torch.no_grad():
            for i in range(0, h - patch_size, stride):
                for j in range(0, w - patch_size, stride):
                    occluded = image.clone()
                    occluded[0, :, i:i+patch_size, j:j+patch_size] = 0
                    
                    output = self.model(occluded)
                    prob = torch.softmax(output, dim=1)[0, target_class].item()
                    
                    diff = baseline_prob - prob
                    attribution[i:i+patch_size, j:j+patch_size] += diff
                    count[i:i+patch_size, j:j+patch_size] += 1
        
        attribution = attribution / (count + 1e-10)
        return self._normalize_attribution(attribution)
