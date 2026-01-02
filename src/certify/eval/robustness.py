#!/usr/bin/env python
"""Robustness evaluation of certified attributions."""

from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from .base import BaseEvaluator


class RobustnessEvaluator(BaseEvaluator):
    """Evaluate robustness via %certified metric."""
    
    def evaluate_batch(
        self,
        cert_results_pkl: Path,
        dataset,
        output_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute %certified aggregated across images.
        
        Args:
            cert_results_pkl: certification results pickle
            dataset: unused (for interface compatibility)
            output_dir: where to save results
        
        Returns:
            results dict: {model -> {method -> {k_percent -> metrics}}}
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load certification results
        cert_results = self.load_cert_results(cert_results_pkl)
        
        results = {}
        
        for model_name in sorted(cert_results.keys()):
            results[model_name] = {}
            methods_dict = cert_results[model_name]
            
            for method_name in sorted(methods_dict.keys()):
                results[model_name][method_name] = {}
                k_dict = methods_dict[method_name]
                
                for k_percent in sorted(k_dict.keys()):
                    entries = k_dict[k_percent]
                    
                    pct_certified = []
                    pct_abstained = []
                    pct_certified_1 = []
                    pct_certified_0 = []
                    certified_radius = []
                    
                    for entry in tqdm(entries, desc=f"{model_name}/{method_name}/k={k_percent}", leave=False):
                        res = entry.get('results', {})
                        c_map = res.get('certified_map')
                        
                        if c_map is None:
                            continue
                        
                        total = c_map.size
                        pct_certified.append(100.0 * np.sum(c_map != -1) / total)
                        pct_abstained.append(100.0 * np.sum(c_map == -1) / total)
                        pct_certified_1.append(100.0 * np.sum(c_map == 1) / total)
                        pct_certified_0.append(100.0 * np.sum(c_map == 0) / total)
                        certified_radius.append(res.get('certified_radius', 0.0))
                    
                    results[model_name][method_name][k_percent] = {
                        'num_images': len(entries),
                        'pct_certified': float(np.mean(pct_certified)) if pct_certified else 0.0,
                        'pct_certified_std': float(np.std(pct_certified)) if len(pct_certified) > 1 else 0.0,
                        'pct_abstained': float(np.mean(pct_abstained)) if pct_abstained else 0.0,
                        'pct_certified_1': float(np.mean(pct_certified_1)) if pct_certified_1 else 0.0,
                        'pct_certified_0': float(np.mean(pct_certified_0)) if pct_certified_0 else 0.0,
                        'certified_radius': float(np.mean(certified_radius)) if certified_radius else 0.0,
                    }
        
        return results
    
    def plot_results(self, results: Dict, output_dir: Path) -> None:
        """
        Plot %certified comparison across methods.
        
        Args:
            results: robustness results dict
            output_dir: where to save figures
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for k_percent in [50, 25, 5]:
            self._plot_robustness_k(results, output_dir, k_percent)
    
    def _plot_robustness_k(self, results: Dict, output_dir: Path, k_percent: int) -> None:
        """Plot robustness at specific K value."""
        models = sorted(results.keys())
        methods = sorted(next(iter(results.values())).keys())
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(methods))
        width = 0.15
        
        for i, model in enumerate(models):
            pct_certs = []
            for method in methods:
                if k_percent in results[model][method]:
                    pct_certs.append(results[model][method][k_percent]['pct_certified'])
                else:
                    pct_certs.append(0.0)
            
            ax.bar(x + i * width, pct_certs, width, label=model)
        
        ax.set_xlabel('Attribution Method', fontsize=12)
        ax.set_ylabel('% Certified Pixels', fontsize=12)
        ax.set_title(f'Certification Robustness (K={k_percent}%)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        out_path = output_dir / f'robustness_k{k_percent}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved robustness figure (K={k_percent}%) to {out_path}")
    
    def plot_stacked_certification(self, results: Dict, output_dir: Path, model_name: str) -> None:
        """
        Plot stacked bar chart with K values grouped, methods as colored bars, 
        showing certified '1', '0', and abstain proportions within each bar.
        
        Args:
            results: robustness results dict {model -> {method -> {k_percent -> metrics}}}
            output_dir: where to save figures
            model_name: which model to plot
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name not in results:
            print(f"Model {model_name} not in results")
            return
        
        model_results = results[model_name]
        methods = sorted(model_results.keys())
        k_values = sorted(model_results[methods[0]].keys(), reverse=True) if model_results else []
        
        # Bright color palette for attribution methods
        method_colors = {
            'GradCAM': '#FF3333',          # Bright red
            'Occlusion': '#FF9900',        # Bright orange
            'IntegratedGradients': '#00CCFF',  # Bright cyan
            'RISE': '#00FF00',             # Bright green
            'LRP': '#FF00FF',              # Bright magenta
        }
        
        # Use method-specific colors or fallback
        colors = [method_colors.get(m, f'C{i}') for i, m in enumerate(methods)]
        
        fig, ax = plt.subplots(figsize=(16, 7))
        
        x = np.arange(len(k_values))
        width = 0.12  # width per method bar within each K group
        
        # For each method, plot grouped bars across K values
        for method_idx, method in enumerate(methods):
            certified_1_list = []
            certified_0_list = []
            abstain_list = []
            
            for k_percent in k_values:
                if k_percent in model_results[method]:
                    metrics = model_results[method][k_percent]
                    certified_1_list.append(metrics.get('pct_certified_1', 0))
                    certified_0_list.append(metrics.get('pct_certified_0', 0))
                    abstain_list.append(metrics.get('pct_abstained', 0))
                else:
                    certified_1_list.append(0)
                    certified_0_list.append(0)
                    abstain_list.append(100)
            
            offset = (method_idx - len(methods) / 2 + 0.5) * width
            
            # Stack: certified "1" (darkest/solid), certified "0" (medium dark), abstain (almost white)
            ax.bar(x + offset, certified_1_list, width, label=method if method_idx == 0 else '', 
                  color=colors[method_idx], alpha=1.0, edgecolor='black', linewidth=0.5)
            ax.bar(x + offset, certified_0_list, width, bottom=certified_1_list, 
                  color=colors[method_idx], alpha=0.65, edgecolor='black', linewidth=0.5)
            ax.bar(x + offset, abstain_list, width, 
                  bottom=np.array(certified_1_list) + np.array(certified_0_list),
                  color='white', alpha=1.0, edgecolor='#CCCCCC', linewidth=0.5)
        
        # Create legend manually
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=methods[i]) 
                          for i in range(len(methods))]
        legend_elements.extend([
            Patch(facecolor='gray', edgecolor='black', label='Certified "1" (Darkest)'),
            Patch(facecolor='gray', alpha=0.65, edgecolor='black', label='Certified "0" (Medium)'),
            Patch(facecolor='white', edgecolor='#CCCCCC', label='Abstain "⊘" (Lightest)'),
        ])
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=10, ncol=2)
        
        ax.set_xlabel('K (%)', fontsize=12)
        ax.set_ylabel('% of pixels', fontsize=12)
        ax.set_title(f'Certification Robustness Breakdown - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{k}' for k in k_values])
        ax.set_ylim([0, 105])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        out_path = output_dir / 'robustness_stacked.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved stacked robustness figure to {out_path}")
