#!/usr/bin/env python
"""Evaluate faithfulness metrics for Experiment 2 (bulk certification).

Inputs:
  outputs/bulk_certifcation/<dataset>/resnet18/results_*.pkl (certification results)
  outputs/checkpoints/<dataset>/resnet18/final_model.pt (trained models)

Outputs (written to outputs/eval/experiment2/faithfulness/):
  <dataset>/resnet18/faithfulness_results.json : per-method per-K faithfulness metrics
  <dataset>/resnet18/figures/ : deletion confidence curves and other plots
"""
import argparse
import glob
import pickle
from pathlib import Path
from collections import defaultdict

import torch

from src.certify.eval.faithfulness import FaithfulnessEvaluator
from src.datasets import get_dataset


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(5):
        if (cur / "src").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def latest_pkl(pkl_paths):
    if not pkl_paths:
        return None
    return max(pkl_paths, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Evaluate faithfulness for Experiment 2 (bulk certification)")
    parser.add_argument(
        "--cert_base",
        default="outputs/bulk_certifcation",
        help="Base directory containing bulk certification pickles",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/eval/experiment2/faithfulness",
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

    repo_root = find_repo_root(Path.cwd())
    cert_base = repo_root / args.cert_base
    output_root = repo_root / args.output_root
    checkpoint_root = repo_root / args.checkpoint_root
    device = args.device

    print(f"Device: {device}")
    print(f"Cert base (bulk): {cert_base}")
    print(f"Output root: {output_root}")
    print(f"Checkpoint root: {checkpoint_root}")

    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        # Bulk certification structure: outputs/bulk_certifcation/<dataset>/resnet18/
        dataset_cert_dir = cert_base / dataset / "resnet18"
        if not dataset_cert_dir.exists():
            print(f"[WARN] No certification dir for {dataset}: {dataset_cert_dir}")
            continue

        checkpoint_base = checkpoint_root / dataset

        # Find latest certification pickle
        pkl_paths = list(dataset_cert_dir.glob("results_*.pkl"))
        pkl_path = latest_pkl(pkl_paths)
        if pkl_path is None:
            print(f"[WARN] No certification pickle found for {dataset} under {dataset_cert_dir}")
            continue

        print(f"  Using certification file: {pkl_path.name}")

        try:
            with open(pkl_path, "rb") as f:
                cert_results = pickle.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read pickle {pkl_path}: {e}")
            continue

        models_available = [m for m in cert_results.keys() if cert_results[m]]
        if not models_available:
            print(f"[WARN] No models with results in {pkl_path.name}")
            continue

        print(f"  Models found: {models_available}")

        # Experiment 2: focus on resnet18
        model_name = "resnet18"
        if model_name not in models_available:
            print(f"[WARN] Model {model_name} not in certification results. Available: {models_available}")
            continue

        # Load dataset
        # IMPORTANT: Use 'val' split to match bulk certification (which certifies on val split)
        print(f"Loading dataset: {dataset}")
        try:
            dataset_obj = get_dataset(dataset, split="val", data_dir=Path("data"))
        except Exception as e:
            print(f"[WARN] Failed to load dataset {dataset}: {e}")
            continue

        print(f"\nEvaluating model: {model_name}")
        output_dir = output_root / dataset / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write temp per-model pickle for evaluator to consume
        pkl_path_temp = output_dir / f"temp_results_{model_name}.pkl"
        pkl_path_temp.parent.mkdir(parents=True, exist_ok=True)
        with open(pkl_path_temp, "wb") as f:
            pickle.dump({model_name: cert_results.get(model_name, {})}, f)

        print(f"  Using pickle: {pkl_path_temp.name}")

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
                cert_results_pkl=pkl_path_temp,
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
                cert_results_pkl=pkl_path_temp,
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
    print("✓ Experiment 2 faithfulness evaluation complete")
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    main()
