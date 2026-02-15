#!/usr/bin/env python
"""Evaluate localization performance (GridPG) for Grid ISIC certification results.

This script:
1. Loads certification results from bulk_certify_grid_isic
2. Computes CertifiedGridPG scores using ground truth masks
3. Generates localization performance plots and saves metrics

Expected inputs:
- Certification results: outputs/bulk_certifcation/grid/isic_2/resnet18/cert_results.pkl
- Grid metadata: data/raw/grid/isic/metadata.json (for GT masks)

Outputs:
- GridPG metrics JSON: outputs/eval/grid/isic/localization_results.json
- Visualization: outputs/eval/grid/isic/localization_gridpg.png
"""

import argparse
import sys
from pathlib import Path
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from certify.eval.localization import LocalizationEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Grid ISIC localization using CertifiedGridPG metric"
    )
    parser.add_argument(
        "--cert_results",
        type=str,
        default="outputs/bulk_certifcation/grid/isic_2/resnet18/cert_results.pkl",
        help="Path to certification results pickle",
    )
    parser.add_argument(
        "--grid_metadata",
        type=str,
        default="data/raw/grid/isic/metadata.json",
        help="Path to grid dataset metadata.json (for ground truth masks)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval/grid/isic",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="Model name (used to access results dict)",
    )
    parser.add_argument(
        "--save_per_k_plots",
        action="store_true",
        help="Save individual plots for each K value",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Convert paths
    cert_results_path = Path(args.cert_results)
    grid_metadata_path = Path(args.grid_metadata)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not cert_results_path.exists():
        print(f"❌ Certification results not found: {cert_results_path}")
        print(f"   Run bulk certification first using run_bulk_certify_grid_isic.py")
        sys.exit(1)
    
    if not grid_metadata_path.exists():
        print(f"❌ Grid metadata not found: {grid_metadata_path}")
        print(f"   Ensure grid dataset was generated properly")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Grid ISIC Localization Evaluation (CertifiedGridPG)")
    print("=" * 80)
    print(f"Certification results: {cert_results_path}")
    print(f"Grid metadata: {grid_metadata_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {args.model_name}")
    print()
    
    # Initialize evaluator
    checkpoint_dir = os.path.join("outputs", "checkpoints", "isic", args.model_name)
    evaluator = LocalizationEvaluator(
        model_name=args.model_name,
        dataset_name="grid_isic",
        checkpoint_dir=checkpoint_dir
    )
    
    # Run evaluation
    print("Computing CertifiedGridPG scores...")
    results = evaluator.evaluate_batch(
        cert_results_pkl=cert_results_path,
        dataset=None,  # Not needed, using metadata
        output_dir=output_dir,
        grid_metadata_json=grid_metadata_path,
    )
    
    if not results:
        print("❌ No localization results computed. Check inputs.")
        sys.exit(1)
    
    # Print summary
    print()
    print("=" * 80)
    print("Localization Results Summary (GridPG Scores)")
    print("=" * 80)
    print(f"{'Method':<20} {'K=50%':<12} {'K=25%':<12} {'K=5%':<12}")
    print("-" * 80)
    
    for method in sorted(results.keys()):
        k50 = results[method].get(50, {}).get('mean_gridpg', 0.0)
        k25 = results[method].get(25, {}).get('mean_gridpg', 0.0)
        k5 = results[method].get(5, {}).get('mean_gridpg', 0.0)
        print(f"{method:<20} {k50:<12.4f} {k25:<12.4f} {k5:<12.4f}")
    
    print("-" * 80)
    print()
    print("Interpretation:")
    print("  GridPG = 1.0  → Perfect localization (all certified pixels in target cell)")
    print("  GridPG = 0.25 → Random (baseline for 2×2 grid)")
    print("  GridPG < 0.25 → Misleading (worse than random)")
    print()
    
    # Save results
    results_json_path = output_dir / "localization_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {results_json_path}")
    
    # Generate plots
    print()
    print("Generating visualizations...")
    evaluator.plot_results(results, output_dir)
    
    # Optional: per-K plots
    if args.save_per_k_plots:
        print("Generating per-K plots...")
        for k in [50, 25, 5]:
            evaluator._plot_localization_k(results, output_dir, k)
    
    print()
    print("=" * 80)
    print("✓ Grid ISIC Localization Evaluation Complete")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - Metrics: localization_results.json")
    print(f"  - Plot: localization_gridpg.png")
    if args.save_per_k_plots:
        print(f"  - Per-K plots: localization_k*.png")
    print()


if __name__ == "__main__":
    main()
