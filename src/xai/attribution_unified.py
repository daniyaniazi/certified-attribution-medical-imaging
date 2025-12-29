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
        self.feature_maps = output.detach().clone()
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach().clone()
    
    def attribute(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """Grad-CAM attribution."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Clone input to create fresh computational graph for each call
        image = image.clone().to(self.device)
        image.requires_grad_(True)
        
        # Reset stored tensors to avoid stale references
        self.feature_maps = None
        self.gradients = None
        
        output = self.model(image)
        logit = output[0, target_class]
        
        self.model.zero_grad()
        if image.grad is not None:
            image.grad.zero_()
        
        # Keep graph around to avoid freed-tensor errors when hooks access saved values
        logit.backward(retain_graph=True)
        
        # Hooks already clone, compute CAM without building new graph
        print(f"[DEBUG] GradCAM fix active: hooks clone tensors")  # TEMPORARY DEBUG
        
        with torch.no_grad():
            gradients = self.gradients
            feature_maps = self.feature_maps
            
            if gradients is None or feature_maps is None:
                # Fallback: return zero heatmap
                h, w = image.shape[-2:]
                return np.zeros((h, w), dtype=np.float32)
            
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * feature_maps, dim=1)[0]
            
            import torch.nn.functional as F
            cam = F.relu(cam)
            
            cam_np = cam.cpu().numpy()
        
        # Clear references to allow garbage collection
        self.feature_maps = None
        self.gradients = None
        
        return self._normalize_attribution(cam_np)


class RISEUnified(AttributionMethod):
    """RISE - unified interface (paper-faithful implementation)."""
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int,
        num_samples: int = 500,
        mask_size: int = 14,
        prob_include: float = 0.5
    ) -> np.ndarray:
        """RISE attribution using logits as per paper specification."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        _, c, h, w = image.shape
        image = image.to(self.device)
        
        attribution = np.zeros((h, w))
        
        import torch.nn.functional as F
        from tqdm import tqdm
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="RISE", disable=True):
                # Generate random mask
                mask_shape = (1, 1, mask_size, mask_size)
                mask = (torch.rand(mask_shape, device=self.device) < prob_include).float()
                
                # Upsample with bilinear interpolation for smooth masks
                mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
                
                # Apply mask and get prediction
                masked_image = image * mask
                
                output = self.model(masked_image)
                # Use LOGIT not probability - critical for proper attribution
                score = output[0, target_class].item()
                
                attribution += score * mask[0, 0].cpu().numpy()
        
        # Normalize by N * p as per RISE paper
        attribution = attribution / (num_samples * prob_include)
        return self._normalize_attribution(attribution)


class OcclusionUnified(AttributionMethod):
    """Occlusion - coarse-grid variant with upsampling."""
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int,
        patch_size: int = 8,
        stride: int = 4
    ) -> np.ndarray:
        """Occlusion attribution using logits (coarse grid with bilinear upsampling)."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        _, c, h, w = image.shape
        image = image.to(self.device)
        
        # Use mean image value as baseline instead of zeros
        baseline_value = image.mean()
        
        with torch.no_grad():
            baseline_output = self.model(image)
            # Use LOGIT not probability - critical for proper attribution
            baseline_score = baseline_output[0, target_class].item()
        
        # Create lower resolution attribution map for efficiency
        h_out = (h - patch_size) // stride + 1
        w_out = (w - patch_size) // stride + 1
        attribution_lr = np.zeros((h_out, w_out))
        
        with torch.no_grad():
            idx = 0
            for i in range(0, h - patch_size + 1, stride):
                jdx = 0
                for j in range(0, w - patch_size + 1, stride):
                    occluded = image.clone()
                    # Use mean value instead of zeros for more natural occlusion
                    occluded[0, :, i:i+patch_size, j:j+patch_size] = baseline_value
                    
                    output = self.model(occluded)
                    # Use LOGIT not probability
                    score = output[0, target_class].item()
                    
                    # Importance = drop in logit when occluded
                    attribution_lr[idx, jdx] = baseline_score - score
                    jdx += 1
                idx += 1
        
        # Upsample attribution map to original image size using bilinear interpolation
        import torch.nn.functional as F
        attribution_tensor = torch.from_numpy(attribution_lr).unsqueeze(0).unsqueeze(0).float()
        attribution_upsampled = F.interpolate(
            attribution_tensor, 
            size=(h, w), 
            mode='bilinear', 
            align_corners=False
        )
        attribution = attribution_upsampled[0, 0].numpy()
        
        return self._normalize_attribution(attribution)


class LRPUnified(AttributionMethod):
    """
    Layer-wise Relevance Propagation (LRP) using Zennit.
    Compatible with ResNet, DenseNet, EfficientNet, MobileNet.
    
    Returns:
        heatmap [H, W] normalized to [0,1]
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        epsilon: float = 1e-6
    ):
        super().__init__(model, device)
        self.epsilon = epsilon

    def attribute(self, image: torch.Tensor, target_class: int) -> np.ndarray:
        """LRP attribution using Zennit's Epsilon rule."""
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)
        image.requires_grad_(True)

        # ---- Zennit imports ----
        try:
            from zennit.composites import EpsilonPlus
            from zennit.attribution import Gradient
        except ImportError:
            raise ImportError(
                "Zennit not installed. Install with: pip install zennit"
            )

        # ---- LRP-ε composite (stable default) ----
        composite = EpsilonPlus(epsilon=self.epsilon)

        # ---- LRP attribution using Gradient attributor ----
        with Gradient(model=self.model, composite=composite) as attributor:
            # Forward pass to get output and relevance
            output, relevance = attributor(image)

        # relevance shape: [1, C, H, W]
        relevance = relevance.detach()

        # ---- Convert to [H, W] heatmap ----
        # Sum over channels and take absolute value
        heatmap = relevance.sum(dim=1)[0]
        heatmap = heatmap.abs().cpu().numpy()

        return self._normalize_attribution(heatmap)
