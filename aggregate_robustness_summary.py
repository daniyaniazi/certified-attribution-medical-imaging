#!/usr/bin/env python
"""Aggregate and plot robustness results across datasets/models for Experiment 1.

Inputs:
  outputs/eval/experiment1/robustness/<dataset>/<model>/robustness_results.json

Outputs (written to outputs/eval/experiment1/robustness/summary/):
  - summary.json : structured aggregates
  - summary.csv  : tabular per-dataset/per-model aggregates
  - README.txt   : quick description and metric definitions

Aggregations:
  * per model (within dataset):
      - mean_pct_certified: weighted by num_images across methods and K
      - entries: total images counted
  * per dataset: weighted mean over models (by entries)
  * overall: weighted mean over all dataset/model entries

Note: This assumes robustness_results.json format produced by RobustnessEvaluator:
  { method -> { k_percent -> { num_images, pct_certified, ... } } }
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

from src.certify.eval.robustness import RobustnessEvaluator


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def weighted_mean(values_and_weights):
    num = 0.0
    den = 0.0
    for v, w in values_and_weights:
        num += v * w
        den += w
    return num / den if den > 0 else 0.0, den


def aggregate_model(result_dict):
    """Aggregate across methods and K for a single model's robustness_results.json."""
    vals = []
    for method, kdict in result_dict.items():
        for k, metrics in kdict.items():
            n = metrics.get("num_images", 0)
            v = metrics.get("pct_certified", 0.0)
            vals.append((v, n))
    mean_cert, total_n = weighted_mean(vals)
    return {
        "mean_pct_certified": mean_cert,
        "total_images": total_n,
    }


def aggregate_dataset(model_aggs):
    vals = [(m["mean_pct_certified"], m["total_images"]) for m in model_aggs]
    mean_cert, total_n = weighted_mean(vals)
    return {
        "mean_pct_certified": mean_cert,
        "total_images": total_n,
    }


def plot_dataset_stacked(dataset_name: str, dataset_dir: Path, checkpoint_root: Path, device: str = "cpu"):
    dataset_results = {}
    model_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and (d / "robustness_results.json").exists()]
    if not model_dirs:
        print(f"[WARN] No model robustness files under {dataset_dir}")
        return

    for mdir in model_dirs:
        model_name = mdir.name
        rr = load_json(mdir / "robustness_results.json")
        dataset_results[model_name] = rr

    first_model = model_dirs[0].name
    evaluator = RobustnessEvaluator(
        dataset_name=dataset_name,
        model_name=first_model,
        checkpoint_dir=checkpoint_root / dataset_name,
        device=device,
    )

    fig_dir = dataset_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    evaluator.plot_stacked_certification(dataset_results, fig_dir, f"{dataset_name}_all_models")
    print(f"  ✓ Stacked plot written to {fig_dir}")


def plot_summary_bar(summary_json: Path, out_dir: Path):
    data = load_json(summary_json)
    per_dataset = data.get("per_dataset", {})
    labels = list(per_dataset.keys())
    vals = [per_dataset[k]["mean_pct_certified"] for k in labels]

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, vals, color="#4C8BF5", edgecolor="black")
    plt.ylabel("Mean % certified")
    plt.ylim(0, 100)
    plt.title("Robustness summary (mean % certified per dataset)")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "summary_mean_pct_certified.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Summary bar written to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot robustness summaries for Experiment 1")
    parser.add_argument(
        "--robustness_root",
        default="outputs/eval/experiment1/robustness",
        help="Root directory containing per-dataset/model robustness results",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="outputs/checkpoints",
        help="Root of checkpoints (dataset subfolders) for plotting convenience",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for evaluator instantiation when plotting (cpu is fine)",
    )
    args = parser.parse_args()

    root = Path(args.robustness_root)
    checkpoint_root = Path(args.checkpoint_root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in root.iterdir() if d.is_dir() and d.name != "summary"]
    summary = {
        "per_dataset": {},
        "per_model": {},
        "overall": {},
    }
    csv_rows = []

    overall_vals = []

    for dpath in datasets:
        dataset_name = dpath.name
        model_dirs = [m for m in dpath.iterdir() if m.is_dir() and (m / "robustness_results.json").exists()]
        model_aggs = []
        for mdir in model_dirs:
            model_name = mdir.name
            rr_path = mdir / "robustness_results.json"
            rr = load_json(rr_path)
            agg = aggregate_model(rr)
            summary["per_model"].setdefault(dataset_name, {})[model_name] = agg
            model_aggs.append(agg)
            csv_rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "mean_pct_certified": agg["mean_pct_certified"],
                "total_images": agg["total_images"],
            })
        if model_aggs:
            d_agg = aggregate_dataset(model_aggs)
            summary["per_dataset"][dataset_name] = d_agg
            overall_vals.append((d_agg["mean_pct_certified"], d_agg["total_images"]))

    # Overall
    mean_overall, total_overall = weighted_mean(overall_vals)
    summary["overall"] = {
        "mean_pct_certified": mean_overall,
        "total_images": total_overall,
    }

    # Save JSON
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    import csv
    csv_path = summary_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "mean_pct_certified", "total_images"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Save README
    readme = """Robustness summary

This directory contains aggregated robustness metrics from outputs/eval/experiment1/robustness.

Fields:
- mean_pct_certified: weighted average of pct_certified across methods and K (weights = num_images per entry)
- total_images: total number of images summed across methods/K for weighting

Files:
- summary.json: per-dataset/per-model aggregates and overall mean
- summary.csv: tabular per-dataset/per-model mean_pct_certified, total_images
"""
    with open(summary_dir / "README.txt", "w") as f:
        f.write(readme)

    print(f"✓ Summary written to {summary_dir}")

    # Plot per-dataset stacked bars and summary bar
    for dpath in datasets:
        dataset_name = dpath.name
        plot_dataset_stacked(dataset_name, dpath, checkpoint_root, device=args.device)

    summary_json = summary_dir / "summary.json"
    if summary_json.exists():
        plot_summary_bar(summary_json, summary_dir / "figures")
    else:
        print(f"[WARN] Missing summary.json at {summary_json}; skipping summary plot")

    print("\n✓ Aggregation and plotting complete")


if __name__ == "__main__":
    main()
