#!/usr/bin/env python
"""Run robustness evaluation for Experiment 2 across datasets (bulk certification).

- Inputs: certification pickles under outputs/bulk_certifcation/<dataset>/resnet18/
- Outputs: robustness JSON + figures under outputs/eval/experiment2/<dataset>/resnet18
- Uses RobustnessEvaluator from src.certify.eval.robustness.
- Model: resnet18 only
"""
import argparse
import sys
from pathlib import Path
import pickle

from src.certify.eval.robustness import RobustnessEvaluator


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


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2 robustness evaluation (bulk certification)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["brain_mri", "chestxray", "fundus", "isic"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--cert_base",
        default="outputs/bulk_certifcation",
        help="Base directory containing bulk certification pickles",
    )
    parser.add_argument(
        "--output_base",
        default="outputs/eval/experiment2",
        help="Directory to write robustness outputs",
    )
    parser.add_argument(
        "--checkpoint_base",
        default="outputs/checkpoints",
        help="Base directory containing checkpoints (passed to evaluator)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for evaluation (auto -> cuda if available else cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = find_repo_root(Path.cwd())
    sys.path.insert(0, str(repo_root))

    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    cert_base = repo_root / args.cert_base
    output_base = repo_root / args.output_base
    checkpoint_root = repo_root / args.checkpoint_base

    print(f"Repo root: {repo_root}")
    print(f"Cert base (bulk): {cert_base}")
    print(f"Output base: {output_base}")
    print(f"Checkpoint base: {checkpoint_root}")
    print(f"Device: {device}")

    for dataset in args.datasets:
        # Bulk certification structure: outputs/bulk_certifcation/<dataset>/resnet18/
        dataset_cert_dir = cert_base / dataset / "resnet18"
        dataset_out_dir = output_base / dataset / "resnet18"
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        # Use dataset-specific checkpoint directory (outputs/checkpoints/<dataset>)
        checkpoint_base_dataset = checkpoint_root / dataset

        # Find latest certification pickle in bulk_certifcation/<dataset>/resnet18/
        pkl_paths = list(dataset_cert_dir.glob("results_*.pkl"))
        pkl_path = latest_pkl(pkl_paths)
        if pkl_path is None:
            print(f"[WARN] No certification pickle found for {dataset} under {dataset_cert_dir}")
            continue

        print(f"\n=== Dataset: {dataset} ===")
        print(f"Using certification file: {pkl_path}")

        with open(pkl_path, "rb") as f:
            cert_results = pickle.load(f)

        # Extract available models from pickle
        models_available = [m for m in cert_results.keys() if cert_results[m]]
        if not models_available:
            print(f"[WARN] No models with results in {pkl_path.name}")
            continue

        print(f"Models found: {models_available}")

        # Experiment 2: focus on resnet18 (if available)
        model_name = "resnet18"
        if model_name not in models_available:
            print(f"[WARN] Model {model_name} not in certification results. Available: {models_available}")
            continue

        print(f"\nEvaluating model: {model_name}")

        evaluator = RobustnessEvaluator(
            dataset_name=dataset,
            model_name=model_name,
            checkpoint_dir=checkpoint_base_dataset,
            device=device,
        )

        rob_results_raw = evaluator.evaluate_batch(
            cert_results_pkl=pkl_path,
            dataset=None,
            output_dir=dataset_out_dir,
        )
        rob_results = rob_results_raw.get(model_name, {})

        evaluator.save_results_json(rob_results, dataset_out_dir / "robustness_results.json")
        evaluator.plot_stacked_certification(
            {model_name: rob_results},
            dataset_out_dir / "figures",
            model_name,
            dataset_name=dataset,
        )

        print(f"Saved: {dataset_out_dir}")

    print("\n✓ Experiment 2 robustness evaluation complete")


if __name__ == "__main__":
    main()
