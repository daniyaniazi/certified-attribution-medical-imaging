"""Sparsification utilities for attribution maps."""
import numpy as np
from typing import Tuple


def sparsify_topk(
    attribution: np.ndarray,
    k_percent: int = 10
) -> np.ndarray:
    """
    Sparsify attribution map by keeping only top K% pixels.
    
    Args:
        attribution: attribution map [H, W] in [0, 1]
        k_percent: percentage of pixels to keep (0-100)
    
    Returns:
        binary mask [H, W] with {0, 1}
    """
    flat = attribution.flatten()
    threshold = np.percentile(flat, 100 - k_percent)
    mask = (attribution >= threshold).astype(np.float32)
    return mask


def sparsify_threshold(
    attribution: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Sparsify attribution map by threshold.
    
    Args:
        attribution: attribution map [H, W] in [0, 1]
        threshold: threshold value
    
    Returns:
        binary mask [H, W] with {0, 1}
    """
    mask = (attribution >= threshold).astype(np.float32)
    return mask


def get_sparsity(mask: np.ndarray) -> float:
    """Get sparsity percentage (% of non-zero elements)."""
    return 100.0 * mask.mean()
