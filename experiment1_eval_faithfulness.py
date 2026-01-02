#!/usr/bin/env python
"""Evaluate faithfulness metrics from certification results for Experiment 1.

Inputs:
  outputs/certifications/<dataset>/<model>/results_*.pkl (certification results)
  outputs/checkpoints/<dataset>/<model>/final_model.pt (trained models)

Outputs (written to outputs/experiment1/faithfulness/):
  <dataset>/<model>/faithfulness_results.json : per-method per-K faithfulness metrics
  <dataset>/<model>/figures/ : deletion confidence curves and other plots
"""
import argparse
import glob
from pathlib import Path
from collections import defaultdict

import torch

from src.certify.eval.faithfulness import FaithfulnessEvaluator
from src.datasets import get_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate faithfulness for Experiment 1")
    parser.add_argument(
        "--certifications_root",
        default="outputs/certifications",
        help="Root directory containing per-dataset/model certification results",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/experiment1/faithfulness",
        help="Root directory where to save faithfulness results",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="outputs/checkpoints",
        help="Root of checkpoints (dataset subfolders)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["brain_mri", "chestxray", "fundus", "isic"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "densenet121", "mobilenet_v2", "efficientnet_b1"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--deletion_steps",
        type=int,
        default=5,
        help="Number of deletion steps for faithfulness evaluation",
    )
    args = parser.parse_args()

    cert_root = Path(args.certifications_root)
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    device = args.device

    print(f"Device: {device}")
    print(f"Certification root: {cert_root}")
    print(f"Output root: {output_root}")

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        dataset_cert_dir = cert_root / dataset
        if not dataset_cert_dir.exists():
            print(f"[WARN] No certification dir for {dataset}: {dataset_cert_dir}")
            continue

        checkpoint_base = checkpoint_root / dataset

        # Find available models with certification results
        model_cert_dirs = {d.name: d for d in dataset_cert_dir.iterdir() if d.is_dir()}
        models_available = list(model_cert_dirs.keys())
        models_to_run = [m for m in args.models if m in models_available]

        if not models_to_run:
            print(f"[WARN] No matching models in {dataset_cert_dir}. Available: {models_available}")
            continue

        # Load dataset
        print(f"Loading dataset: {dataset}")
        try:
            dataset_obj = get_dataset(dataset, split="test", data_dir=Path("data"))
        except Exception as e:
            print(f"[WARN] Failed to load dataset {dataset}: {e}")
            continue

        for model_name in models_to_run:
            print(f"\nEvaluating model: {model_name}")
            model_cert_dir = model_cert_dirs[model_name]
            output_dir = output_root / dataset / model_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Find latest certification pickle
            pkl_files = sorted(model_cert_dir.glob("results_*.pkl"))
            if not pkl_files:
                print(f"[WARN] No certification pickles found in {model_cert_dir}")
                continue
            pkl_path = pkl_files[-1]
            print(f"  Using pickle: {pkl_path.name}")

            # Initialize evaluator
            try:
                evaluator = FaithfulnessEvaluator(
                    dataset_name=dataset,
                    model_name=model_name,
                    checkpoint_dir=checkpoint_base,
                    device=device,
                )
            except Exception as e:
                print(f"[WARN] Failed to initialize evaluator for {model_name}: {e}")
                continue

            # Evaluate faithfulness
            try:
                fa_results = evaluator.evaluate_batch(
                    cert_results_pkl=pkl_path,
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    deletion_steps=args.deletion_steps,
                )

                # Save results
                evaluator.save_results_json(fa_results, output_dir / "faithfulness_results.json")
                
                # Plot results
                fig_dir = output_dir / "figures"
                fig_dir.mkdir(parents=True, exist_ok=True)
                evaluator.plot_results(fa_results, fig_dir)
                
                # Plot deletion confidence curves
                evaluator.plot_deletion_confidence_curves(
                    cert_results_pkl=pkl_path,
                    dataset=dataset_obj,
                    output_dir=fig_dir,
                    deletion_steps=args.deletion_steps - 1,
                )

                print(f"  ✓ Saved faithfulness results to {output_dir}")
                print(f"  ✓ Methods: {list(fa_results.keys())}")

            except Exception as e:
                print(f"[ERROR] Failed to evaluate faithfulness for {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*60}")
    print("✓ Experiment 1 faithfulness evaluation complete")
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    main()
