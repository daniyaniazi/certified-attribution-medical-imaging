"""
Randomized smoothing for certified pixel attribution.

Based on paper approach:
  Eq. (4): Sparsification h_K(x) = top-K pixels of heatmap
  Eq. (5): Certification via majority voting over noisy samples
  Eq. (7): Certified radius R = σ * Φ^(-1)(τ)

References:
  - Eq. (4): Sparsification equation
  - Eq. (5): Certification via aggregation
  - Eq. (7): Robustness radius from segmentation smoothing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import ndtri  # For inverse CDF: Φ^(-1)
from typing import Tuple, Callable, Dict
from tqdm import tqdm
import time


class RandomizedSmoothingAttributor:
    """
    Randomized smoothing for certified pixel attribution.
    
    Based on paper approach:
    - Add Gaussian noise N(0, σ²I) to input
    - Compute attribution multiple times
    - Aggregate via majority voting on each pixel
    """
    
    def __init__(
        self,
        model: nn.Module,
        attribution_func: Callable,
        device: str = 'cpu'
    ):
        """
        Args:
            model: PyTorch model
            attribution_func: callable that takes (image) -> attribution [H, W]
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.attribution_func = attribution_func
        self.device = device
        self.model.eval()
    
    def certify(
        self,
        image: torch.Tensor,
        k_percent: int = 30,
        target_class: int = 1,
        sigma: float = 0.15,
        num_samples: int = 100,
        tau: float = 0.75,
        batch_size: int = 16,
        alpha: float = 0.001
    ) -> Dict:
        """
        Certify pixel attribution via randomized smoothing (Paper Eq. 5-7).
        
        Algorithm:
        1. For each of n samples:
           - Sample noise ε_t ~ N(0, σ²I)
           - Compute noisy image: x_t = x + ε_t
           - Get attribution heatmap: heat_t = h(x_t)
           - Sparsify: mask_t = h_K(x_t)  [top-K% pixels]
        
        2. For each pixel i, compute:
           - p_1[i] = P(h_K(x+ε)=1) via majority voting
           - p_0[i] = P(h_K(x+ε)=0) = 1 - p_1[i]
        
        3. Certify with threshold τ:
           - If p_1[i] > τ → cert[i] = 1
           - Else if p_0[i] > τ → cert[i] = 0
           - Else → cert[i] = abstain
        
        4. Compute certified radius:
           R = σ * Φ^(-1)(τ)
           Guarantees: for ||δ||_2 < R, certified pixels unchanged
        
        Args:
            image: input image [C, H, W] or [1, C, H, W]
            k_percent: sparsification level (keep top K%)
            target_class: target class for attribution
            sigma: Gaussian noise std (paper: 0.15)
            num_samples: number of smoothing samples (paper: 100)
            tau: certification threshold (paper: 0.75)
            batch_size: batch size for efficiency
            alpha: significance level for confidence intervals
        
        Returns:
            dict with:
            - certified_map: {-1 (abstain), 0, 1} [H, W]
            - p_1: probability estimates [H, W]
            - p_0: probability estimates [H, W]
            - pct_certified: % of certified pixels
            - pct_abstained: % of abstained pixels
            - certified_radius: R = σ * Φ^(-1)(τ)
            - stats: voting statistics
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        _, c, h, w = image.shape
        image = image.to(self.device)
        
        # Eq. (5): Initialize vote counters for per-pixel probabilities
        count_class_1 = np.zeros((h, w), dtype=np.int32)  # count where h_K = 1
        count_class_0 = np.zeros((h, w), dtype=np.int32)  # count where h_K = 0
        
        print(f"\n[Randomized Smoothing] σ={sigma}, τ={tau}, n={num_samples}, K={k_percent}%")
        
        # Step 1-2: Sample noise and aggregate votes
        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Smoothing samples", disable=False):
                batch_end = min(i + batch_size, num_samples)
                batch_n = batch_end - i
                
                # Sample noise: ε_t ~ N(0, σ²I)
                noise = torch.randn(batch_n, c, h, w, device=self.device) * sigma
                
                # Create noisy images: x_t = x + ε_t
                noisy_images = image + noise
                noisy_images = torch.clamp(noisy_images, 0, 1)
                
                # Process each noisy sample
                for j in range(batch_n):
                    # Compute attribution: heat_t = h(x_t)
                    heat = self.attribution_func(noisy_images[j:j+1], target_class)
                    
                    # Sparsify: h_K(x_t) = top-K% pixels (Eq. 4)
                    mask = self._sparsify_topk(heat, k_percent)
                    
                    # Aggregate: count votes
                    count_class_1 += (mask > 0.5).astype(np.int32)
                    count_class_0 += (mask < 0.5).astype(np.int32)
        
        # Step 2: Compute per-pixel probabilities (Eq. 5)
        p_1 = count_class_1 / num_samples  # P(h_K(x+ε) = 1)
        p_0 = count_class_0 / num_samples  # P(h_K(x+ε) = 0)
        
        # Step 3: Certification with threshold τ (Eq. 5)
        certified_map = np.full((h, w), -1, dtype=np.int8)  # -1 = abstain (⊘)
        
        certified_1 = 0
        certified_0 = 0
        abstained = 0
        
        for i in range(h):
            for j in range(w):
                if p_1[i, j] >= tau:
                    certified_map[i, j] = 1
                    certified_1 += 1
                elif p_0[i, j] >= tau:
                    certified_map[i, j] = 0
                    certified_0 += 1
                else:
                    abstained += 1
        
        total_pixels = h * w
        certified_pixels = certified_1 + certified_0
        
        # Step 4: Compute certified radius (Eq. 7 - from Segmentation Smoothing)
        # R = σ * Φ^(-1)(τ)
        # where Φ^(-1) is inverse normal CDF
        if tau >= 0.5 and tau < 1.0:
            radius = sigma * ndtri(tau)
        else:
            # Invalid tau or tau=1 (no robustness)
            radius = 0.0
        
        # Compilation of results
        results = {
            'certified_map': certified_map,
            'p_1': p_1,  # Probability estimates for class 1
            'p_0': p_0,  # Probability estimates for class 0
            'pct_certified': 100.0 * certified_pixels / total_pixels,
            'pct_abstained': 100.0 * abstained / total_pixels,
            'pct_certified_1': 100.0 * certified_1 / total_pixels,
            'pct_certified_0': 100.0 * certified_0 / total_pixels,
            'certified_radius': float(radius),  # R = σ * Φ^(-1)(τ)
            'stats': {
                'sigma': sigma,
                'tau': tau,
                'k_percent': k_percent,
                'num_samples': num_samples,
                'certified_1': int(certified_1),
                'certified_0': int(certified_0),
                'abstained': int(abstained),
                'total_pixels': int(total_pixels),
                'certified_radius': float(radius)
            }
        }
        
        # Print summary
        print(f"[Results] Certified: {results['pct_certified']:.1f}% | "
              f"Abstained: {results['pct_abstained']:.1f}% | "
              f"Radius: {radius:.4f}")
        
        return results
    
    def _sparsify_topk(self, heatmap: np.ndarray, k_percent: int) -> np.ndarray:
        """
        Sparsify heatmap to binary mask keeping top-K% (Eq. 4).
        
        Args:
            heatmap: [H, W] normalized to [0, 1]
            k_percent: percentage of pixels to keep
        
        Returns:
            binary mask [H, W] with {0, 1}
        """
        flat = heatmap.flatten()
        threshold = np.percentile(flat, 100 - k_percent)
        mask = (heatmap >= threshold).astype(np.float32)
        return mask
    
    @staticmethod
    def compute_abstention_rate(certified_map: np.ndarray) -> float:
        """Compute % of abstained pixels."""
        abstained = np.sum(certified_map == -1)
        total = certified_map.size
        return 100.0 * abstained / total
