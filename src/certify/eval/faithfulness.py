#!/usr/bin/env python
"""Faithfulness evaluation of certified attributions."""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from .base import BaseEvaluator


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluate faithfulness via deletion-based analysis."""
    
    def evaluate_batch(
        self,
        cert_results_pkl: Path,
        dataset,
        output_dir: Path,
        deletion_steps: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness for all images, methods, K values.
        
        Args:
            cert_results_pkl: certification results pickle
            dataset: PyTorch dataset with [idx] -> {'image': tensor, 'label': int}
            output_dir: where to save results
            deletion_steps: number of deletion steps per K
        
        Returns:
            results dict: {method_name -> {k_percent -> metrics}}
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load certification results
        cert_results = self.load_cert_results(cert_results_pkl)
        
        results = defaultdict(lambda: defaultdict(dict))
        
        methods = sorted(cert_results[self.model_name].keys())
        k_values = [50, 25, 5]
        
        for method_name in methods:
            print(f"\n  Evaluating {method_name}...")
            
            for k in k_values:
                if k not in cert_results[self.model_name][method_name]:
                    continue
                
                entries = cert_results[self.model_name][method_name][k]
                aucs = []
                baseline_confs = []
                loc_fracs = []
                
                for entry in tqdm(entries, desc=f"    K={k}%", leave=False):
                    img_idx = entry.get('image_idx')
                    label = entry.get('label')
                    
                    # Load image from dataset
                    sample = dataset[img_idx]
                    image = sample['image'] if isinstance(sample, dict) else sample[0]
                    image = image.cpu().numpy() if torch.is_tensor(image) else image
                    
                    # Load certified maps from entry results
                    certified_maps = self._extract_certified_maps(entry, k)
                    if not certified_maps:
                        continue
                    
                    # Compute faithfulness
                    head_id = entry.get('head_id')
                    target_cell = entry.get('target_cell')
                    scale = entry.get('scale')
                    auc, baseline_conf, loc_frac = self._compute_deletion_auc(
                        image,
                        certified_maps,
                        label,
                        deletion_steps=deletion_steps,
                        head_id=head_id,
                        target_cell=target_cell,
                        scale=scale,
                    )
                    
                    aucs.append(auc)
                    baseline_confs.append(baseline_conf)
                    if loc_frac is not None:
                        loc_fracs.append(loc_frac)
                
                # Aggregate
                if aucs:
                    entry_res = {
                        'mean_auc': float(np.mean(aucs)),
                        'std_auc': float(np.std(aucs)),
                        'num_images': len(aucs),
                        'mean_baseline_conf': float(np.mean(baseline_confs)),
                    }
                    if loc_fracs:
                        entry_res['mean_loc_cert1_frac'] = float(np.mean(loc_fracs))
                        entry_res['std_loc_cert1_frac'] = float(np.std(loc_fracs))
                    results[method_name][k] = entry_res
        
        return dict(results)
    
    def _extract_certified_maps(
        self,
        entry: Dict,
        k_percent: int
    ) -> Dict[int, np.ndarray]:
        """Extract certified maps from entry."""
        res = entry.get('results', {})
        c_map = res.get('certified_map')
        
        if c_map is None:
            return {}
        
        return {k_percent: c_map}
    
    def _compute_deletion_auc(
        self,
        image: np.ndarray,
        certified_maps: Dict[int, np.ndarray],
        target_class: int,
        deletion_steps: int = 5,
        head_id: Optional[int] = None,
        target_cell: Optional[int] = None,
        scale: Optional[int] = None,
    ) -> Tuple[float, float, Optional[float]]:
        """
        Compute AUC for deletion-based faithfulness.
        
        Args:
            image: [C, H, W] in [0, 1]
            certified_maps: {K -> certified_map [-1/0/1]}
            target_class: ground truth class
            deletion_steps: number of deletion steps
        
        Returns:
            (auc, baseline_confidence)
        """
        def _forward_conf(img_np: np.ndarray) -> float:
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(img_tensor) if head_id is None else self.model(img_tensor, head_id=head_id)
                probs = torch.softmax(out, dim=1)
            return probs[0, target_class].item()

        baseline_conf = _forward_conf(image)
        
        # Collect all certified-1 pixels across K values
        all_certified_pixels = []
        for k in sorted(certified_maps.keys()):
            cmap = certified_maps[k]
            ys, xs = np.where(cmap == 1)
            all_certified_pixels.extend(list(zip(ys, xs)))
        
        if not all_certified_pixels:
            return 0.0, baseline_conf, None
        
        # Iteratively delete pixels
        deleted_img = image.copy()
        deletion_confs = []

        for step in range(1, deletion_steps + 1):
            num_to_delete = len(all_certified_pixels) // deletion_steps
            pixels_to_delete = all_certified_pixels[(step - 1) * num_to_delete:step * num_to_delete]
            
            for y, x in pixels_to_delete:
                deleted_img[:, y, x] = 0.0
            
            del_conf = _forward_conf(deleted_img)
            deletion_confs.append(del_conf)
        
        # Compute AUC
        drops = [baseline_conf - c for c in deletion_confs]
        auc = float(np.mean([d / baseline_conf if baseline_conf > 0 else 0.0 for d in drops]))

        # Localization metric: fraction of certified-1 pixels inside target cell (if provided)
        loc_frac = None
        if target_cell is not None and scale is not None:
            h, w = certified_maps[next(iter(certified_maps))].shape
            cell_h = h // int(scale)
            cell_w = w // int(scale)
            row = int(target_cell) // int(scale)
            col = int(target_cell) % int(scale)
            y0, x0 = row * cell_h, col * cell_w
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y0:y0 + cell_h, x0:x0 + cell_w] = 1
            # Use largest-K certified map (max key)
            c_map = certified_maps[max(certified_maps.keys())]
            cert1 = (c_map == 1)
            total_cert1 = cert1.sum()
            if total_cert1 > 0:
                inside = (cert1 & (mask == 1)).sum()
                loc_frac = float(inside / total_cert1)

        return auc, baseline_conf, loc_frac
    
    def plot_results(self, results: Dict, output_dir: Path) -> None:
        """
        Plot deletion-based faithfulness curves.
        
        Args:
            results: faithfulness results dict
            output_dir: where to save figures
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Bar chart: Methods vs AUC
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = sorted(results.keys())
        k_values = sorted(next(iter(results.values())).keys())
        
        x = np.arange(len(methods))
        width = 0.25
        
        for i, k in enumerate(k_values):
            aucs = [results[m][k]['mean_auc'] if k in results[m] else 0.0
                   for m in methods]
            ax.bar(x + i * width, aucs, width, label=f'K={k}%')
        
        ax.set_xlabel('Attribution Method', fontsize=12)
        ax.set_ylabel('Faithfulness (AUC)', fontsize=12)
        ax.set_title(f'Deletion-Based Faithfulness - {self.model_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        out_path = output_dir / 'faithfulness_comparison.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved faithfulness figure to {out_path}")

    # ------------------------------------------------------------------
    # Paper-style confidence curves (Figure 8): confidence vs deletion steps
    # ------------------------------------------------------------------
    def plot_deletion_confidence_curves(
        self,
        cert_results_pkl: Path,
        dataset,
        output_dir: Path,
        deletion_steps: int = 4,
        reuse_existing: bool = True,
        data_json: Optional[Path] = None,
        save_json: bool = True,
    ) -> None:
        """
        Plot mean GT confidence as certified-1 pixels are progressively deleted.

        Args:
            cert_results_pkl: certification results pickle
            dataset: dataset providing images by index
            output_dir: base dir to save figures
            deletion_steps: number of fractional deletion steps (default 4 → 0,25,50,75,100%)
            reuse_existing: if True and cached JSON exists, load instead of recomputing
            data_json: optional path to cached stats JSON (defaults to output_dir/faithfulness_confidence_curves_data.json)
            save_json: whether to persist computed stats to JSON
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        if data_json is None:
            data_json = output_dir / 'faithfulness_confidence_curves_data.json'

        cert_results = self.load_cert_results(cert_results_pkl)

        if self.model_name not in cert_results:
            print(f"Model {self.model_name} not in certification results")
            return

        methods = sorted(cert_results[self.model_name].keys())
        k_values = sorted({k for m in methods for k in cert_results[self.model_name][m].keys()})
        if not k_values:
            print("No K values available; skipping confidence curves")
            return

        def _forward_conf(img_np: np.ndarray, target_class: int) -> float:
            t = torch.from_numpy(img_np).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(t)
                probs = torch.softmax(out, dim=1)
            return probs[0, target_class].item()

        def _compute_conf_curves(entries):
            steps = np.linspace(0.0, 1.0, deletion_steps + 1)
            curves = []
            for entry in entries:
                img_idx = entry.get('image_idx')
                label = entry.get('label')
                sample = dataset[img_idx]
                image = sample['image'] if isinstance(sample, dict) else sample[0]
                image = image.cpu().numpy() if torch.is_tensor(image) else image
                res = entry.get('results', {})
                c_map = res.get('certified_map')
                if c_map is None:
                    continue
                ys, xs = np.where(c_map == 1)
                coords = list(zip(ys, xs))
                if not coords:
                    continue
                baseline = _forward_conf(image, label)
                img_mod = image.copy()
                confs = [baseline]
                for frac in steps[1:]:
                    num = int(len(coords) * frac)
                    for (y, x) in coords[:num]:
                        img_mod[:, y, x] = 0.0
                    confs.append(_forward_conf(img_mod, label))
                curves.append(confs)
            if not curves:
                return steps, None
            curves = np.array(curves)
            return steps, (curves.mean(axis=0), curves.std(axis=0), curves.shape[0])

        def _compute_all_stats():
            data = {
                'model': self.model_name,
                'deletion_steps': deletion_steps,
                'step_fracs': None,
                'k_values': {},
            }
            for k_val in k_values:
                k_entry = {'methods': {}}
                for method in methods:
                    entries = cert_results[self.model_name][method].get(k_val)
                    if not entries:
                        continue
                    steps, stats = _compute_conf_curves(entries)
                    if stats is None:
                        continue
                    mean_conf, std_conf, n_imgs = stats
                    if data['step_fracs'] is None:
                        data['step_fracs'] = steps.tolist()
                    k_entry['methods'][method] = {
                        'mean_conf': mean_conf.tolist(),
                        'std_conf': std_conf.tolist(),
                        'num_images': int(n_imgs),
                    }
                if k_entry['methods']:
                    data['k_values'][str(k_val)] = k_entry
            return data

        data = None
        if reuse_existing and data_json.exists():
            try:
                with open(data_json, 'r') as f:
                    cached = json.load(f)
                if cached.get('deletion_steps') == deletion_steps:
                    data = cached
                else:
                    print("Cached deletion_steps mismatch; recomputing")
            except Exception:
                print("Failed to load cached confidence data; recomputing")

        if data is None:
            data = _compute_all_stats()
            if save_json:
                with open(data_json, 'w') as f:
                    json.dump(data, f)
                print(f"  ✓ Saved deletion-step confidence data to {data_json}")

        if not data.get('k_values'):
            print("No confidence data to plot")
            return

        fig, axes = plt.subplots(
            1, len(data['k_values']), figsize=(6 * len(data['k_values']), 6), sharey=True
        )
        if len(data['k_values']) == 1:
            axes = [axes]

        step_fracs = data.get('step_fracs', [])
        step_labels = ['Orig'] + [f"del {int(frac*100)}%" for frac in step_fracs[1:]] if step_fracs else []

        for ax, (k_str, k_entry) in zip(axes, sorted(data['k_values'].items(), key=lambda kv: float(kv[0]))):
            for method, stats in k_entry['methods'].items():
                ax.plot(step_labels, stats['mean_conf'], marker='o', label=method)
            ax.set_xlabel('Deletion steps')
            ax.set_title(f'K={k_str}%')
            ax.set_ylim(bottom=0.0)
            ax.grid(alpha=0.3)

        axes[0].set_ylabel('GT class confidence')
        legend_handles, legend_labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)
        if legend_handles:
            axes[0].legend(legend_handles, legend_labels, loc='upper left')
        fig.suptitle(f'Deletion-Step Confidence - {self.model_name}', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = output_dir / 'faithfulness_confidence_curves.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved deletion-step confidence curves to {out_path}")
