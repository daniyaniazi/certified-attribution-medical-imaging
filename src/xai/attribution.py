"""Attribution methods: Integrated Gradients, Grad-CAM, RISE, Occlusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
from typing import Callable, Tuple
from tqdm import tqdm


class IntegratedGradients:
    """Integrated Gradients attribution method."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.eval()
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int,
        num_steps: int = 50,
        baseline: torch.Tensor = None
    ) -> np.ndarray:
        """
        Compute Integrated Gradients.
        
        Args:
            image: input image [C, H, W] or [1, C, H, W]
            target_class: class index for attribution
            num_steps: number of integration steps
            baseline: baseline image (default: all zeros)
        
        Returns:
            attribution map [H, W]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        if baseline is None:
            baseline = torch.zeros_like(image)
        
        baseline = baseline.to(self.device)
        image = image.to(self.device)
        
        # Generate interpolated images
        alphas = torch.linspace(0, 1, num_steps + 1, device=self.device)
        
        accumulated_grads = None
        
        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            logit = output[0, target_class]
            
            # Backward pass
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
        
        # Average accumulated gradients
        avg_grads = accumulated_grads / len(alphas)
        
        # Multiply by input delta
        integrated_grads = (image - baseline) * avg_grads
        
        # Sum over channels and normalize
        attribution = torch.sum(integrated_grads, dim=1)[0]
        attribution = attribution.abs().detach().cpu().numpy()
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-10)
        
        return attribution


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.model.eval()
        
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_features)
        self.target_layer.register_backward_hook(self._save_gradients)
    
    def _save_features(self, module, input, output):
        self.feature_maps = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """
        Compute Grad-CAM attribution.
        
        Args:
            image: input image [C, H, W] or [1, C, H, W]
            target_class: class index for attribution
        
        Returns:
            attribution map [H, W]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        image.requires_grad_(True)
        
        # Forward
        output = self.model(image)
        logit = output[0, target_class]
        
        # Backward
        self.model.zero_grad()
        logit.backward(retain_graph=False)
        
        # Compute Grad-CAM
        gradients = self.gradients  # [1, C, H, W]
        feature_maps = self.feature_maps  # [1, C, H, W]
        
        # Average gradient over spatial dimensions
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted sum of feature maps
        cam = torch.sum(weights * feature_maps, dim=1)[0]  # [H, W]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam_np = cam.detach().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-10)
        
        return cam_np


class RISE:
    """RISE: Randomized Input Sampling for Explanation."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.eval()
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int,
        num_samples: int = 1000,
        mask_size: int = 7,
        prob_include: float = 0.5
    ) -> np.ndarray:
        """
        Compute RISE attribution.
        
        Args:
            image: input image [C, H, W] or [1, C, H, W]
            target_class: class index for attribution
            num_samples: number of random masks to sample
            mask_size: size of mask grid
            prob_include: probability of including each mask region
        
        Returns:
            attribution map [H, W]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        _, c, h, w = image.shape
        image = image.to(self.device)
        
        # Initialize attribution map
        attribution = np.zeros((h, w))
        
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="RISE", disable=False):
                # Random mask
                mask_shape = (1, 1, mask_size, mask_size)
                mask = (torch.rand(mask_shape) < prob_include).float()
                
                # Upsample mask to image size
                mask = F.interpolate(
                    mask,
                    size=(h, w),
                    mode='nearest'
                )
                
                # Apply mask
                masked_image = image * mask
                
                # Forward
                output = self.model(masked_image)
                prob = torch.softmax(output, dim=1)[0, target_class].item()
                
                # Accumulate
                attribution += prob * mask[0, 0].cpu().numpy()
        
        # Normalize
        attribution = attribution / num_samples
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-10)
        
        return attribution


class Occlusion:
    """Occlusion-based attribution (perturbation method)."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.eval()
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: int,
        patch_size: int = 16,
        stride: int = 8
    ) -> np.ndarray:
        """
        Compute occlusion-based attribution.
        
        Args:
            image: input image [C, H, W] or [1, C, H, W]
            target_class: class index for attribution
            patch_size: size of occlusion patch
            stride: stride for sliding window
        
        Returns:
            attribution map [H, W]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        _, c, h, w = image.shape
        image = image.to(self.device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(image)
            baseline_prob = torch.softmax(baseline_output, dim=1)[0, target_class].item()
        
        # Occlusion map
        attribution = np.zeros((h, w))
        count = np.zeros((h, w))
        
        with torch.no_grad():
            for i in range(0, h - patch_size, stride):
                for j in range(0, w - patch_size, stride):
                    # Create occluded image
                    occluded = image.clone()
                    occluded[0, :, i:i+patch_size, j:j+patch_size] = 0
                    
                    # Forward
                    output = self.model(occluded)
                    prob = torch.softmax(output, dim=1)[0, target_class].item()
                    
                    # Difference
                    diff = baseline_prob - prob
                    
                    # Accumulate (importance = drop in probability)
                    attribution[i:i+patch_size, j:j+patch_size] += diff
                    count[i:i+patch_size, j:j+patch_size] += 1
        
        # Average overlapping regions
        attribution = attribution / (count + 1e-10)
        
        # Normalize
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-10)
        
        return attribution
