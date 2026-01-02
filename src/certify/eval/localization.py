#!/usr/bin/env python
"""Localization evaluation of certified attributions."""

from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
import json

import numpy as np
import matplotlib.pyplot as plt

from .base import BaseEvaluator


class LocalizationEvaluator(BaseEvaluator):
    """Evaluate localization via CertifiedGridPG metric."""
    
    @staticmethod
    def compute_certified_gridpg(certified_map: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Compute CertifiedGridPG score.
        
        Args:
            certified_map: {1, 0, ⊘} array [H, W] from certification
            gt_mask: binary mask [H, W] where 1 = correct region, 0 = elsewhere
        
        Returns:
            GridPG score = #(certified-1 pixels in mask) / #(all certified-1 pixels)
        
        Interpretation:
            - 1.0 = perfect localization (all certified pixels in correct region)
            - 0.25 (2×2 grid) = random
            - < random = misleading attribution
        """
        # Find all certified-1 pixels
        certified_pixels = (certified_map == 1)
        
        if not np.any(certified_pixels):
            # No certified pixels; return 0 (no signal)
            return 0.0
        
        # Find certified pixels inside the mask
        certified_in_mask = np.logical_and(certified_pixels, gt_mask == 1)
        
        # Score
        gridpg = np.sum(certified_in_mask) / np.sum(certified_pixels)
        return float(gridpg)
    
    def evaluate_batch(
        self,
        cert_results_pkl: Path,
        dataset,
        output_dir: Path,
        gt_masks: Dict[int, np.ndarray] = None,
        grid_metadata_json: Path = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute CertifiedGridPG using ground truth masks.
        
        Args:
            cert_results_pkl: certification results pickle
            dataset: unused (for interface compatibility)
            output_dir: where to save results
            gt_masks: {image_idx -> binary mask [H, W]} (optional for manual masks)
            grid_metadata_json: path to metadata.json from grid generation (auto-builds masks)
        
        Returns:
            results dict: {method -> {k_percent -> metrics}}
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build masks from grid metadata if provided
        if grid_metadata_json is not None:
            gt_masks = self._build_grid_masks_from_metadata(grid_metadata_json)
            print(f"  ✓ Built {len(gt_masks)} masks from grid metadata")
        
        if gt_masks is None or not gt_masks:
            print(f"  ⚠️  No ground truth masks provided. Skipping localization eval.")
            return {}
        
        # Load certification results
        cert_results = self.load_cert_results(cert_results_pkl)
        
        results = defaultdict(lambda: defaultdict(dict))
        
        methods = sorted(cert_results[self.model_name].keys())
        k_values = [50, 25, 5]
        
        for method_name in methods:
            print(f"  Evaluating {method_name}...")
            
            for k in k_values:
                if k not in cert_results[self.model_name][method_name]:
                    continue
                
                entries = cert_results[self.model_name][method_name][k]
                gridpg_scores = []
                
                for entry in entries:
                    img_idx = entry.get('image_idx')
                    res = entry.get('results', {})
                    c_map = res.get('certified_map')
                    
                    if c_map is None or img_idx not in gt_masks:
                        continue
                    
                    mask = gt_masks[img_idx]
                    gridpg = self.compute_certified_gridpg(c_map, mask)
                    gridpg_scores.append(gridpg)
                
                # Aggregate
                if gridpg_scores:
                    results[method_name][k] = {
                        'mean_gridpg': float(np.mean(gridpg_scores)),
                        'std_gridpg': float(np.std(gridpg_scores)),
                        'num_images': len(gridpg_scores),
                    }
        
        return dict(results)
    
    def plot_results(self, results: Dict, output_dir: Path) -> None:
        """
        Plot CertifiedGridPG comparison - single grouped bar chart.
        
        Args:
            results: localization results dict
            output_dir: where to save figures
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self._plot_localization_grouped(results, output_dir)
    
    @staticmethod
    def _build_grid_masks_from_metadata(grid_metadata_json: Path) -> Dict[int, np.ndarray]:
        """
        Build binary masks for grid target cells from metadata.json.
        
        Args:
            grid_metadata_json: path to metadata.json from grid generation
        
        Returns:
            {grid_image_idx -> binary mask [H*scale, W*scale]}
        """
        with open(grid_metadata_json) as f:
            metadata = json.load(f)
        
        scale = metadata['scale']
        target_cell = metadata['target_cell']
        img_h, img_w = metadata['image_size']
        num_grids = len(metadata.get('target_classes', []))
        
        masks = {}
        
        for grid_idx in range(num_grids):
            # Create binary mask for target cell
            full_h = img_h * scale
            full_w = img_w * scale
            mask = np.zeros((full_h, full_w), dtype=np.uint8)
            
            # Compute target cell position
            target_row = target_cell // scale
            target_col = target_cell % scale
            y = target_row * img_h
            x = target_col * img_w
            
            # Set target cell region to 1
            mask[y:y+img_h, x:x+img_w] = 1
            
            masks[grid_idx] = mask
        
        return masks
    
    def _plot_localization_grouped(self, results: Dict, output_dir: Path) -> None:
        """Plot grouped bar chart: K values on x-axis, methods as colored bars."""
        methods = sorted(results.keys())
        k_values = [50, 25, 5]
        
        # Prepare data
        data = {method: [] for method in methods}
        for method in methods:
            for k in k_values:
                if k in results[method]:
                    data[method].append(results[method][k]['mean_gridpg'])
                else:
                    data[method].append(0.0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(k_values))
        width = 0.15
        colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c', '#34495e', '#f39c12']
        
        for i, method in enumerate(methods):
            offset = (i - len(methods) / 2) * width
            ax.bar(x + offset, data[method], width, label=method, color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Sparsification parameter (K)', fontsize=12)
        ax.set_ylabel('Certified GridPG Score', fontsize=12)
        ax.set_title('Certified Localization across Attribution Methods', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{k}%' for k in k_values])
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random (2×2 grid)')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        out_path = output_dir / 'localization_gridpg.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved localization figure to {out_path}")
    
    def _plot_localization_k(self, results: Dict, output_dir: Path, k_percent: int) -> None:
        """Plot localization at specific K value."""
        methods = sorted(results.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(methods))
        gridpqs = [results[m][k_percent]['mean_gridpg'] if k_percent in results[m] else 0.0
                  for m in methods]
        
        ax.bar(x, gridpqs, color='steelblue', alpha=0.7)
        
        ax.set_xlabel('Attribution Method', fontsize=12)
        ax.set_ylabel('Certified GridPG Score', fontsize=12)
        ax.set_title(f'Localization Performance (K={k_percent}%)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        out_path = output_dir / f'localization_k{k_percent}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved localization figure (K={k_percent}%) to {out_path}")
