"""Evaluation metrics for certified attributions."""
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.ndimage import binary_dilation
import cv2


class CertificationEvaluator:
    """Evaluate certified attribution maps."""
    
    @staticmethod
    def compute_faithfulness_deletion(
        model,
        image: np.ndarray,
        attribution: np.ndarray,
        target_class: int,
        device: str = 'cpu',
        num_steps: int = 50
    ) -> Tuple[list, float]:
        """
        Deletion curve: % accuracy drop as high-attribution pixels are removed.
        
        Args:
            model: PyTorch model
            image: input image [C, H, W] in [0, 1]
            attribution: attribution map [H, W] in [0, 1]
            target_class: target class
            device: 'cpu' or 'cuda'
            num_steps: number of deletion steps
        
        Returns:
            (deletion_scores, auc) where deletion_scores is [0, 1] for each step
        """
        import torch
        
        # Get baseline prediction
        img_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            baseline_output = model(img_tensor)
            baseline_conf = torch.softmax(baseline_output, dim=1)[0, target_class].item()
        
        # Sort pixels by importance
        flat_attr = attribution.flatten()
        sorted_indices = np.argsort(-flat_attr)  # Descending
        
        h, w = attribution.shape
        deletion_scores = []
        
        for step in range(num_steps):
            # Delete top-attribution pixels
            pct = (step + 1) / num_steps
            num_to_delete = int(np.ceil(h * w * pct))
            
            # Create deleted image
            deleted_img = image.copy()
            deleted_indices = sorted_indices[:num_to_delete]
            
            for idx in deleted_indices:
                i, j = np.unravel_index(idx, (h, w))
                deleted_img[:, i, j] = 0  # Zero out
            
            # Get confidence after deletion
            deleted_tensor = torch.from_numpy(deleted_img).unsqueeze(0).to(device)
            with torch.no_grad():
                deleted_output = model(deleted_tensor)
                deleted_conf = torch.softmax(deleted_output, dim=1)[0, target_class].item()
            
            deletion_scores.append(deleted_conf)
        
        # Compute AUC
        auc = np.mean([1 - s for s in deletion_scores])
        
        return deletion_scores, auc
    
    @staticmethod
    def compute_localization_accuracy(
        attribution: np.ndarray,
        mask: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Localization accuracy: IoU between top-attribution region and ground truth mask.
        
        Args:
            attribution: attribution map [H, W] in [0, 1]
            mask: ground truth binary mask [H, W]
            threshold: threshold for binarizing attribution
        
        Returns:
            IoU score [0, 1]
        """
        attr_binary = (attribution > threshold).astype(np.float32)
        
        intersection = np.sum(attr_binary * mask)
        union = np.sum(attr_binary) + np.sum(mask) - intersection
        
        iou = intersection / (union + 1e-10)
        return iou
    
    @staticmethod
    def compute_sensitivity_specificity(
        attribution: np.ndarray,
        mask: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[float, float]:
        """
        Sensitivity (True Positive Rate) and Specificity (True Negative Rate).
        
        Args:
            attribution: attribution map [H, W] in [0, 1]
            mask: ground truth binary mask [H, W]
            threshold: threshold for binarizing attribution
        
        Returns:
            (sensitivity, specificity)
        """
        attr_binary = (attribution > threshold).astype(np.int32)
        mask_binary = mask.astype(np.int32)
        
        tp = np.sum((attr_binary == 1) & (mask_binary == 1))
        tn = np.sum((attr_binary == 0) & (mask_binary == 0))
        fp = np.sum((attr_binary == 1) & (mask_binary == 0))
        fn = np.sum((attr_binary == 0) & (mask_binary == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return sensitivity, specificity
    
    @staticmethod
    def evaluate_certified(results: Dict) -> Dict[str, float]:
        """
        Evaluate certified attribution map from smoothing results.
        
        Args:
            results: output dict from RandomizedSmoothingAttributor.certify()
                     contains: certified_map, p_1, p_0, stats, certified_radius
        
        Returns:
            dict of metrics and analysis
        """
        certified_map = results['certified_map']
        p_1 = results['p_1']
        p_0 = results['p_0']
        
        total_pixels = certified_map.size
        certified = np.sum(certified_map != -1)
        abstained = np.sum(certified_map == -1)
        certified_to_1 = np.sum(certified_map == 1)
        certified_to_0 = np.sum(certified_map == 0)
        
        # Confidence analysis
        certified_1_confidences = p_1[certified_map == 1]
        certified_0_confidences = p_0[certified_map == 0]
        
        metrics = {
            'pct_certified': 100.0 * certified / total_pixels,
            'pct_abstained': 100.0 * abstained / total_pixels,
            'pct_certified_1': 100.0 * certified_to_1 / total_pixels,
            'pct_certified_0': 100.0 * certified_to_0 / total_pixels,
            'certified_radius': results.get('certified_radius', 0.0),
            'avg_confidence_class_1': float(np.mean(certified_1_confidences)) if len(certified_1_confidences) > 0 else 0.0,
            'avg_confidence_class_0': float(np.mean(certified_0_confidences)) if len(certified_0_confidences) > 0 else 0.0,
        }
        
        return metrics

    @staticmethod
    def compute_percent_certified(certified_map: np.ndarray) -> float:
        """Fraction of pixels that are certified (non-abstain)."""
        total = certified_map.size
        certified = np.sum(certified_map != -1)
        return certified / total if total > 0 else 0.0

    @staticmethod
    def compute_certified_gridpg(
        certified_map: np.ndarray,
        m: int,
        cell_row: int,
        cell_col: int
    ) -> float:
        """
        Certified Grid Pointing Game (GridPG) score.

        Splits the map into an m x m grid of equal cells and computes
        the ratio of certified-1 pixels within the selected cell to the
        total certified-1 pixels across the entire grid.
        """
        h, w = certified_map.shape
        cell_h = h // m
        cell_w = w // m

        y0 = cell_row * cell_h
        x0 = cell_col * cell_w
        y1 = y0 + cell_h if cell_row < m - 1 else h
        x1 = x0 + cell_w if cell_col < m - 1 else w

        cell = certified_map[y0:y1, x0:x1]
        total_pos = np.sum(certified_map == 1)
        cell_pos = np.sum(cell == 1)

        return (cell_pos / total_pos) if total_pos > 0 else 0.0

    @staticmethod
    def compute_certified_gridpg_mask(
        certified_map: np.ndarray,
        subimage_mask: np.ndarray
    ) -> float:
        """
        Certified GridPG using an explicit ground-truth subimage mask.

        Args:
            certified_map: {-1,0,1} array [H, W]
            subimage_mask: boolean or {0,1} array [H, W] where True/1 indicates
                           pixels belonging to the ground-truth subimage.

        Returns:
            Ratio of certified-1 pixels inside the subimage to total certified-1
            pixels across the whole grid.
        """
        mask_bool = subimage_mask.astype(bool)
        total_pos = np.sum(certified_map == 1)
        cell_pos = np.sum((certified_map == 1) & mask_bool)
        return (cell_pos / total_pos) if total_pos > 0 else 0.0

    @staticmethod
    def compute_certified_gridpg_bbox(
        certified_map: np.ndarray,
        y0: int,
        x0: int,
        y1: int,
        x1: int
    ) -> float:
        """
        Certified GridPG using an explicit bounding box for the subimage.
        """
        h, w = certified_map.shape
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        x0 = max(0, min(x0, w))
        x1 = max(0, min(x1, w))
        cell = certified_map[y0:y1, x0:x1]
        total_pos = np.sum(certified_map == 1)
        cell_pos = np.sum(cell == 1)
        return (cell_pos / total_pos) if total_pos > 0 else 0.0

    @staticmethod
    def compute_certified_deletion(
        model,
        image: np.ndarray,
        certified_maps: Dict[int, np.ndarray],
        k_values: list,
        target_class: int,
        device: str = 'cpu'
    ) -> Tuple[list, float]:
        """
        Deletion-based faithfulness for certified maps.

        Iteratively remove pixels certified as 1, starting from lowest K,
        and record the confidence for the target class after each removal.
        Returns deletion curve and AUC (mean drop proxy).
        """
        import torch

        img_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            base_out = model(img_tensor)
            base_conf = torch.softmax(base_out, dim=1)[0, target_class].item()

        deleted = image.copy()
        deletion_scores = []

        for k in sorted(k_values):
            if k not in certified_maps:
                continue
            cmap = certified_maps[k]
            ys, xs = np.where(cmap == 1)
            for y, x in zip(ys, xs):
                deleted[:, y, x] = 0.0
            del_tensor = torch.from_numpy(deleted).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(del_tensor)
                conf = torch.softmax(out, dim=1)[0, target_class].item()
            deletion_scores.append(conf)

        # AUC proxy: mean normalized drop relative to baseline
        if len(deletion_scores) == 0:
            auc = 0.0
        else:
            drops = [(base_conf - s) for s in deletion_scores]
            # Normalize by baseline to be in [0,1]
            norm_drops = [d / base_conf if base_conf > 0 else 0.0 for d in drops]
            auc = float(np.mean(norm_drops))

        return deletion_scores, auc
