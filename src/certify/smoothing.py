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
from scipy.stats import beta as beta_dist  # Clopper-Pearson intervals
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
        alpha: float = 0.001,
        save_noisy_samples: bool = True,
        max_noisy_samples: int = 3
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
              - p_0[i] = P(h_K(x+ε)=0) via majority voting
              - Use Clopper-Pearson lower confidence bounds with significance α
        
          3. Certify with threshold τ using confidence bounds:
              - If LB(p_1[i]) ≥ τ → cert[i] = 1
              - Else if LB(p_0[i]) ≥ τ → cert[i] = 0
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
            - heatmap_clean: attribution on clean image
            - ss_map: mean of sparsified noisy masks (Smoothed Sparsified)
            - certified_masks: dict of binary masks for certified 1/0/abstain
            - sample_noisy_heatmaps: up to max_noisy_samples raw noisy heatmaps
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        _, c, h, w = image.shape
        image = image.to(self.device)

        def to_numpy_2d(arr: np.ndarray) -> np.ndarray:
            arr_np = np.array(arr)
            while arr_np.ndim > 2:
                if arr_np.shape[0] == 1:
                    arr_np = arr_np.squeeze(0)
                elif arr_np.shape[-1] == 1:
                    arr_np = arr_np.squeeze(-1)
                else:
                    # Average over channels if still multi-channel
                    if arr_np.ndim == 3:
                        arr_np = arr_np.mean(axis=0)
                    elif arr_np.ndim == 4:
                        arr_np = arr_np.mean(axis=(0, 1))
                    break
            if arr_np.ndim != 2:
                raise ValueError(f'Cannot reduce attribution to 2D, shape: {arr_np.shape}')
            return arr_np.astype(np.float32)
        
        # Attribution on the clean image for visualization/storage
        with torch.enable_grad():
            heat_clean = self.attribution_func(image, target_class)
        heat_clean_np = to_numpy_2d(heat_clean)
        
        # Eq. (5): Initialize vote counters for per-pixel probabilities
        count_class_1 = np.zeros((h, w), dtype=np.int32)  # count where h_K = 1
        count_class_0 = np.zeros((h, w), dtype=np.int32)  # count where h_K = 0
        sum_masks = np.zeros((h, w), dtype=np.float32)    # for SS (mean of masks)
        sample_noisy_heatmaps = []
        
        print(f"\n[Randomized Smoothing] σ={sigma}, τ={tau}, n={num_samples}, K={k_percent}%")
        
        # Step 1-2: Sample noise and aggregate votes
        # IMPORTANT: Keep gradients enabled for gradient-based attribution methods (IG, GradCAM, LRP)
        # torch.no_grad() would break these methods!
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
                # Must keep gradients enabled for gradient-based methods!
                with torch.enable_grad():
                    heat = self.attribution_func(noisy_images[j:j+1], target_class)
                heat_np = to_numpy_2d(heat)

                # Some attribution methods (e.g., GradCAM) return low-res maps; resize to input HxW
                if heat_np.shape != (h, w):
                    heat_t = torch.from_numpy(heat_np).float().unsqueeze(0).unsqueeze(0)
                    heat_t = F.interpolate(heat_t, size=(h, w), mode="bilinear", align_corners=False)
                    heat_np = heat_t.squeeze().cpu().numpy()
                
                # Sparsify: h_K(x_t) = top-K% pixels (Eq. 4)
                mask = self._sparsify_topk(heat_np, k_percent)
                sum_masks += mask
                if save_noisy_samples and len(sample_noisy_heatmaps) < max_noisy_samples:
                    sample_noisy_heatmaps.append(heat_np)
                
                # Aggregate: count votes
                count_class_1 += (mask > 0.5).astype(np.int32)
                count_class_0 += (mask < 0.5).astype(np.int32)
        
        # Step 2: Compute per-pixel probabilities (Eq. 5)
        p_1 = count_class_1 / num_samples  # P(h_K(x+ε) = 1)
        p_0 = count_class_0 / num_samples  # P(h_K(x+ε) = 0)
        ss_map = sum_masks / num_samples   # Smoothed Sparsified average mask

        # Clopper-Pearson lower bounds for p_1 and p_0 at significance α
        # LB = Beta(alpha; k, n-k+1)
        # Ensure arrays are float for beta ppf
        k1 = count_class_1.astype(np.int64)
        k0 = count_class_0.astype(np.int64)
        n = int(num_samples)

        # Avoid invalid parameters for beta.ppf
        # When k=0 → LB=0; when k=n → LB=1
        p1_lower = np.where(
            k1 == 0, 0.0,
            np.where(k1 == n, 1.0, beta_dist.ppf(alpha, k1, n - k1 + 1))
        )
        p0_lower = np.where(
            k0 == 0, 0.0,
            np.where(k0 == n, 1.0, beta_dist.ppf(alpha, k0, n - k0 + 1))
        )
        
        # Step 3: Certification with threshold τ (Eq. 5)
        certified_map = np.full((h, w), -1, dtype=np.int8)  # -1 = abstain (⊘)
        
        certified_1 = 0
        certified_0 = 0
        abstained = 0
        
        for i in range(h):
            for j in range(w):
                # Use lower confidence bounds for certification decisions
                if p1_lower[i, j] >= tau:
                    certified_map[i, j] = 1
                    certified_1 += 1
                elif p0_lower[i, j] >= tau:
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
            'p_1': p_1,  # Probability estimates for class 1 (MLE)
            'p_0': p_0,  # Probability estimates for class 0 (MLE)
            'p1_lower': p1_lower,  # Clopper-Pearson lower bounds for class 1
            'p0_lower': p0_lower,  # Clopper-Pearson lower bounds for class 0
            'pct_certified': 100.0 * certified_pixels / total_pixels,
            'pct_abstained': 100.0 * abstained / total_pixels,
            'pct_certified_1': 100.0 * certified_1 / total_pixels,
            'pct_certified_0': 100.0 * certified_0 / total_pixels,
            'certified_radius': float(radius),  # R = σ * Φ^(-1)(τ)
            'heatmap_clean': heat_clean_np,
            'ss_map': ss_map,
            'certified_masks': {
                'certified_1': (certified_map == 1).astype(np.uint8),
                'certified_0': (certified_map == 0).astype(np.uint8),
                'abstain': (certified_map == -1).astype(np.uint8),
            },
            'sample_noisy_heatmaps': sample_noisy_heatmaps,
            'stats': {
                'sigma': sigma,
                'tau': tau,
                'k_percent': k_percent,
                'num_samples': num_samples,
                'alpha': alpha,
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
