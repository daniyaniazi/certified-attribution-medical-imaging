#!/usr/bin/env python
"""Run robustness evaluation for Experiment 1 across datasets/models.

- Inputs: certification pickles under outputs/certifications/<dataset>/
- Outputs: robustness JSON + figures under outputs/eval/experiment1/robustness/<dataset>/<model>
- Uses RobustnessEvaluator from src.certify.eval.robustness.
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
    parser = argparse.ArgumentParser(description="Experiment 1 robustness evaluation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["brain_mri", "chestxray", "fundus", "isic"],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["densenet121", "mobilenet_v2", "efficientnet_b1", "resnet18"],
        help="Model names to include (filtered from cert results)",
    )
    parser.add_argument(
        "--cert_base",
        default="outputs/certifications",
        help="Base directory containing certification pickles",
    )
    parser.add_argument(
        "--output_base",
        default="outputs/eval/experiment1/robustness",
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
    checkpoint_base = repo_root / args.checkpoint_base

    print(f"Repo root: {repo_root}")
    print(f"Cert base: {cert_base}")
    print(f"Output base: {output_base}")
    print(f"Checkpoint base: {checkpoint_base}")
    print(f"Device: {device}")

    for dataset in args.datasets:
        dataset_cert_dir = cert_base / dataset
        dataset_out_dir = output_base / dataset
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        pkl_paths = list(dataset_cert_dir.rglob("*.pkl"))
        pkl_path = latest_pkl(pkl_paths)
        if pkl_path is None:
            print(f"[WARN] No certification pickle found for {dataset} under {dataset_cert_dir}")
            continue

        print(f"\n=== Dataset: {dataset} ===")
        print(f"Using certification file: {pkl_path}")

        with open(pkl_path, "rb") as f:
            cert_results = pickle.load(f)

        models_available = [m for m in cert_results.keys() if cert_results[m]]
        models_to_run = [m for m in models_available if m in args.models]
        if not models_to_run:
            print(f"[WARN] No matching models in cert file. Available: {models_available}")
            continue

        dataset_all_results = {}

        for model_name in models_to_run:
            print(f"\nEvaluating model: {model_name}")
            model_out_dir = dataset_out_dir / model_name
            model_out_dir.mkdir(parents=True, exist_ok=True)

            evaluator = RobustnessEvaluator(
                dataset_name=dataset,
                model_name=model_name,
                checkpoint_dir=checkpoint_base,
                device=device,
            )

            rob_results_raw = evaluator.evaluate_batch(
                cert_results_pkl=pkl_path,
                dataset=None,
                output_dir=model_out_dir,
            )
            rob_results = rob_results_raw.get(model_name, {})

            evaluator.save_results_json(rob_results, model_out_dir / "robustness_results.json")
            evaluator.plot_stacked_certification({model_name: rob_results}, model_out_dir / "figures", model_name)

            dataset_all_results[model_name] = rob_results
            print(f"Saved: {model_out_dir}")

        # Combined stacked bars across models for this dataset
        if dataset_all_results:
            combined_fig_dir = dataset_out_dir / "figures"
            combined_fig_dir.mkdir(parents=True, exist_ok=True)
            # Reuse evaluator from last model (only needs plotting)
            last_model = models_to_run[-1]
            evaluator = RobustnessEvaluator(
                dataset_name=dataset,
                model_name=last_model,
                checkpoint_dir=checkpoint_base,
                device=device,
            )
            evaluator.plot_stacked_certification(dataset_all_results, combined_fig_dir, f"{dataset}_all_models")
            print(f"Combined figures: {combined_fig_dir}")

    print("\n✓ Experiment 1 robustness evaluation complete")


if __name__ == "__main__":
    main()
