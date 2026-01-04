#!/usr/bin/env python
"""Aggregate and plot faithfulness results across datasets for Experiment 2.

Inputs:
  outputs/eval/experiment2/faithfulness/<dataset>/resnet18/faithfulness_results.json

Outputs (written to outputs/eval/experiment2/faithfulness/summary/):
  - summary.json : structured aggregates
  - summary.csv  : tabular per-dataset aggregates
  - README.txt   : quick description and metric definitions
  - figures/     : aggregated visualizations
"""
import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from src.certify.eval.faithfulness import FaithfulnessEvaluator

# Colorblind-friendly palette (Okabe-Ito inspired) used consistently across plots
METHOD_COLORS = {
    "IntegratedGradients": "#0072B2",  # blue
    "GradCAM": "#D55E00",              # vermilion
    "RISE": "#009E73",                # bluish green
    "Occlusion": "#E69F00",           # orange
    "LRP": "#CC79A7",                 # reddish purple
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def weighted_mean(values_and_weights):
    """Compute weighted mean. Returns (mean, total_weight)."""
    num = 0.0
    den = 0.0
    for v, w in values_and_weights:
        num += v * w
        den += w
    return num / den if den > 0 else 0.0, den


def aggregate_dataset(result_dict):
    """Aggregate across methods and K for a dataset's faithfulness_results.json."""
    vals = []
    for method, kdict in result_dict.items():
        for k, metrics in kdict.items():
            n = metrics.get("num_images", 1)
            auc = metrics.get("mean_auc", 0.0)
            vals.append((auc, n))
    mean_auc, total_n = weighted_mean(vals)
    return {
        "mean_auc": mean_auc,
        "total_images": total_n,
    }


def aggregate_across_datasets(dataset_results_dict):
    """Aggregate faithfulness results across datasets, preserving method/K structure.
    
    Args:
        dataset_results_dict: {dataset_name: {method: {k: metrics}}}
    
    Returns:
        {method: {k: {mean metrics}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for dataset_name, methods_dict in dataset_results_dict.items():
        for method, k_dict in methods_dict.items():
            for k, metrics in k_dict.items():
                for metric_name, value in metrics.items():
                    if metric_name == "num_images":
                        aggregated[method][k]["num_images"].append(value)
                    else:
                        n = metrics.get("num_images", 1)
                        aggregated[method][k][metric_name].append((value, n))
    
    # Compute means
    result = {}
    for method, k_dict in aggregated.items():
        result[method] = {}
        for k, metric_dict in k_dict.items():
            result[method][k] = {
                "num_images": sum(metric_dict["num_images"]),
            }
            for metric_name, weighted_vals in metric_dict.items():
                if metric_name != "num_images":
                    mean_val, _ = weighted_mean(weighted_vals)
                    result[method][k][metric_name] = mean_val
    
    return result


def plot_auc_curves(aggregated_results, output_path: Path, title_suffix: str, deletion_steps: int = 4):
    """Plot AUC curves for aggregated faithfulness results across deletion steps."""
    methods = sorted(aggregated_results.keys())
    if not methods:
        print(f"[WARN] No methods found for {title_suffix}; skipping AUC plot")
        return
    
    colors = [METHOD_COLORS.get(m, f'C{i}') for i, m in enumerate(methods)]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for method_idx, method in enumerate(methods):
        k_values = sorted(aggregated_results[method].keys())
        auc_vals = []
        
        for k_percent in k_values:
            metrics = aggregated_results[method][k_percent]
            auc_vals.append(metrics.get("mean_auc", 0.0))
        
        ax.plot(
            range(len(k_values)),
            auc_vals,
            marker="o",
            label=method,
            color=colors[method_idx],
            linewidth=2.5,
            markersize=8,
        )
    
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f'K={k}%' for k in k_values], fontsize=11)
    ax.set_xlabel('Sparsification Level (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean AUC', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title(f'Faithfulness (AUC) {title_suffix}', fontsize=13, fontweight='bold')
    ax.legend(title='Attribution Method', fontsize=9, ncol=2)
    ax.grid(axis='both', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ AUC curves plot: {output_path}")


def plot_dataset_stacked(dataset_name: str, dataset_dir: Path):
    """Plot per-dataset faithfulness (single model - resnet18) with stacked visualization."""
    result_file = dataset_dir / "resnet18" / "faithfulness_results.json"
    if not result_file.exists():
        print(f"[WARN] No results found for {dataset_name} at {result_file}")
        return

    data = load_json(result_file)
    
    fig_dir = dataset_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot AUC curves for this dataset
    plot_auc_curves(
        data,
        fig_dir / "dataset_avg_faithfulness.png",
        f"(Dataset: {dataset_name.upper()}, Model: resnet18)"
    )
    
    # Also save to dataset root for easy access
    plot_auc_curves(
        data,
        dataset_dir / "avg_faithfulness.png",
        f"(Dataset: {dataset_name.upper()}, Model: resnet18)"
    )


def aggregate_by_method(root: Path):
    """Aggregate mean_auc per attribution method across all datasets."""
    method_vals = defaultdict(list)
    for dpath in root.iterdir():
        if not dpath.is_dir() or dpath.name == "summary":
            continue
        result_file = dpath / "resnet18" / "faithfulness_results.json"
        if not result_file.exists():
            continue
        
        data = load_json(result_file)
        for method, kdict in data.items():
            for k, metrics in kdict.items():
                n = metrics.get("num_images", 1)
                auc = metrics.get("mean_auc", 0.0)
                method_vals[method].append((auc, n))
    
    method_aggs = {}
    for method, vals in method_vals.items():
        mean_auc, total_n = weighted_mean(vals)
        method_aggs[method] = {
            "mean_auc": mean_auc,
            "total_images": total_n,
        }
    return method_aggs


def plot_global_method_bar(method_aggs: dict, out_dir: Path):
    """Plot bar chart of avg mean_auc per attribution method across all experiments."""
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(method_aggs.keys())
    vals = [method_aggs[m]["mean_auc"] for m in methods]
    
    colors = [METHOD_COLORS.get(m, f'C{i}') for i, m in enumerate(methods)]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, vals, color=colors, edgecolor="black", linewidth=1.2)
    plt.ylabel("Mean AUC", fontsize=12, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.title("Overall attribution method faithfulness (all datasets, resnet18)")
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "global_method_faithfulness.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Global method bar written to {out_path}")


def aggregate_by_dataset_and_method(root: Path):
    """Aggregate mean_auc per method for each dataset."""
    dataset_method = defaultdict(lambda: defaultdict(list))
    for dpath in root.iterdir():
        if not dpath.is_dir() or dpath.name == "summary":
            continue
        dataset_name = dpath.name
        result_file = dpath / "resnet18" / "faithfulness_results.json"
        if not result_file.exists():
            continue
        
        data = load_json(result_file)
        for method, kdict in data.items():
            for k, metrics in kdict.items():
                n = metrics.get("num_images", 1)
                auc = metrics.get("mean_auc", 0.0)
                dataset_method[dataset_name][method].append((auc, n))
    
    dataset_method_aggs = {}
    for dataset_name, method_vals in dataset_method.items():
        dataset_method_aggs[dataset_name] = {}
        for method, vals in method_vals.items():
            mean_auc, total_n = weighted_mean(vals)
            dataset_method_aggs[dataset_name][method] = {
                "mean_auc": mean_auc,
                "total_images": total_n,
            }
    return dataset_method_aggs


def plot_method_by_dataset(dataset_method_aggs: dict, out_dir: Path):
    """Single figure: grouped bars per dataset showing mean AUC per attribution method."""
    if not dataset_method_aggs:
        print("[WARN] No dataset-method aggregates to plot")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(dataset_method_aggs.keys())
    methods = sorted({m for d in datasets for m in dataset_method_aggs[d].keys()})
    x = np.arange(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(16, 7))

    for idx, method in enumerate(methods):
        vals = [dataset_method_aggs[d].get(method, {}).get("mean_auc", 0.0) for d in datasets]
        offset = (idx - len(methods) / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=method, color=METHOD_COLORS.get(method, f'C{idx}'))

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=11)
    ax.set_ylabel("Mean AUC", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title("Attribution faithfulness by dataset (mean AUC, resnet18)", fontsize=13, fontweight="bold")
    ax.legend(title="Attribution method", ncol=3, fontsize=9, loc='upper right')
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    out_path = out_dir / "method_by_dataset.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Method-by-dataset comparison written to {out_path}")


def plot_summary_bar(summary_json: Path, out_dir: Path):
    """Plot summary bar of mean AUC per dataset."""
    data = load_json(summary_json)
    per_dataset = data.get("per_dataset", {})
    labels = list(per_dataset.keys())
    vals = [per_dataset[k]["mean_auc"] for k in labels]

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, vals, color="#4C8BF5", edgecolor="black")
    plt.ylabel("Mean AUC", fontsize=12, fontweight="bold")
    plt.ylim(0, 1.0)
    plt.title("Faithfulness summary (mean AUC per dataset, resnet18)")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "summary_mean_auc.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Summary bar written to {out_path}")


def aggregate_confidence_curves(root: Path):
    """Aggregate confidence curves per dataset and overall from cached JSON files."""
    dataset_curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    overall_curves = defaultdict(lambda: defaultdict(list))
    step_fracs = []
    found_any = False
    
    for dpath in root.iterdir():
        if not dpath.is_dir() or dpath.name == "summary":
            continue
        dataset_name = dpath.name
        
        # Load confidence curve data from resnet18 model's figures directory
        conf_curve_file = dpath / "resnet18" / "figures" / "faithfulness_confidence_curves_data.json"
        if not conf_curve_file.exists():
            print(f"[WARN] No confidence curves data for {dataset_name} at {conf_curve_file}")
            continue
        
        try:
            conf_data = load_json(conf_curve_file)
            step_fracs = conf_data.get("step_fracs", [])
            found_any = True
            
            for method, k_dict in conf_data.items():
                if method == "step_fracs":
                    continue
                for k_val, curve in k_dict.items():
                    dataset_curves[dataset_name][method][k_val].append(curve)
                    overall_curves[method][k_val].append(curve)
        except Exception as e:
            print(f"[WARN] Failed to load confidence curves for {dataset_name}: {e}")
            continue
    
    # Average curves per dataset
    dataset_avg = {}
    for dataset_name, method_dict in dataset_curves.items():
        dataset_avg[dataset_name] = {}
        for method, k_dict in method_dict.items():
            dataset_avg[dataset_name][method] = {}
            for k_val, curves_list in k_dict.items():
                dataset_avg[dataset_name][method][k_val] = np.mean(curves_list, axis=0).tolist()
    
    # Average curves overall
    overall_avg = {"step_fracs": step_fracs} if found_any else {}
    for method, k_dict in overall_curves.items():
        overall_avg[method] = {}
        for k_val, curves_list in k_dict.items():
            overall_avg[method][k_val] = np.mean(curves_list, axis=0).tolist()
    
    return dataset_avg, overall_avg, step_fracs


def plot_confidence_curves(curves_dict: dict, step_fracs: list, output_path: Path, title: str):
    """Plot GT confidence vs deletion steps (Figure 8 style)."""
    methods_order = ["IntegratedGradients", "GradCAM", "RISE", "Occlusion", "LRP"]
    k_values = sorted({k for m in curves_dict.keys() if m in methods_order for k in curves_dict[m].keys()})
    
    if not k_values:
        print(f"[WARN] No K values found for confidence curves in {title}")
        return
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 6), sharey=True)
    if len(k_values) == 1:
        axes = [axes]
    
    step_labels = ['Orig'] + [f"{int(frac*100)}%" for frac in step_fracs[1:]] if step_fracs else []
    
    for ax_idx, k_val in enumerate(k_values):
        ax = axes[ax_idx]
        for method in methods_order:
            if method in curves_dict and k_val in curves_dict[method]:
                curve = curves_dict[method][k_val]
                ax.plot(range(len(curve)), curve, marker='o', label=method, 
                       color=METHOD_COLORS.get(method, 'gray'), linewidth=2, markersize=6)
        
        ax.set_xlabel('Deletion step', fontsize=11, fontweight='bold')
        ax.set_title(f'K={k_val}%', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(step_labels)))
        ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel('GT class confidence', fontsize=12, fontweight='bold')
    
    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(methods_order), fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confidence curves: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate and plot faithfulness summaries for Experiment 2")
    parser.add_argument(
        "--faithfulness_root",
        default="outputs/eval/experiment2/faithfulness",
        help="Root directory containing per-dataset faithfulness results",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for evaluator instantiation (cpu is fine)",
    )
    args = parser.parse_args()

    root = Path(args.faithfulness_root)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d for d in root.iterdir() if d.is_dir() and d.name != "summary"]
    summary = {
        "per_dataset": {},
        "overall": {},
    }
    csv_rows = []

    overall_vals = []

    for dpath in datasets:
        dataset_name = dpath.name
        result_file = dpath / "resnet18" / "faithfulness_results.json"
        if not result_file.exists():
            print(f"[WARN] No results for {dataset_name}")
            continue

        data = load_json(result_file)
        agg = aggregate_dataset(data)
        summary["per_dataset"][dataset_name] = agg
        overall_vals.append((agg["mean_auc"], agg["total_images"]))

        csv_rows.append({
            "dataset": dataset_name,
            "model": "resnet18",
            "mean_auc": agg["mean_auc"],
            "total_images": agg["total_images"],
        })

    # Overall
    mean_overall, total_overall = weighted_mean(overall_vals)
    summary["overall"] = {
        "mean_auc": mean_overall,
        "total_images": total_overall,
    }

    # Save JSON
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    csv_rows.sort(key=lambda r: r["dataset"])
    csv_path = summary_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "mean_auc", "total_images"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Save README
    readme = """Faithfulness summary (Experiment 2)

This directory contains aggregated faithfulness metrics from outputs/eval/experiment2/faithfulness.

Fields:
- mean_auc: weighted average of mean_auc across methods and K (weights = num_images per entry)
- total_images: total number of images summed across methods/K for weighting

Files:
- summary.json: per-dataset aggregates and overall mean
- summary.csv: tabular per-dataset mean_auc, total_images
"""
    with open(summary_dir / "README.txt", "w") as f:
        f.write(readme)

    print(f"✓ Summary written to {summary_dir}")

    # Plot per-dataset faithfulness
    dataset_results = {}
    for dpath in datasets:
        dataset_name = dpath.name
        plot_dataset_stacked(dataset_name, dpath)
        
        result_file = dpath / "resnet18" / "faithfulness_results.json"
        if result_file.exists():
            dataset_results[dataset_name] = load_json(result_file)

    # Overall averaged AUC curves across all datasets
    if dataset_results:
        aggregated = aggregate_across_datasets(dataset_results)
        plot_auc_curves(
            aggregated,
            summary_dir / "figures" / "overall_avg_faithfulness.png",
            "(Overall, all datasets, resnet18)"
        )
    
    # Global method faithfulness (per method across all datasets)
    method_aggs = aggregate_by_method(root)
    plot_global_method_bar(method_aggs, summary_dir / "figures")

    # Method faithfulness by dataset
    dataset_method_aggs = aggregate_by_dataset_and_method(root)
    plot_method_by_dataset(dataset_method_aggs, summary_dir / "figures")

    # Summary bar
    summary_json = summary_dir / "summary.json"
    if summary_json.exists():
        plot_summary_bar(summary_json, summary_dir / "figures")
    else:
        print("[WARN] summary.json not found; skipping summary bar chart")

    # Generate confidence curves (Figure 8 style)
    print("\n" + "="*60)
    print("Generating GT confidence curves (Figure 8 style)")
    print("="*60)
    dataset_conf_avg, overall_conf_avg, step_fracs = aggregate_confidence_curves(root)
    
    print(f"[DEBUG] Found {len(dataset_conf_avg)} datasets with confidence curves")
    print(f"[DEBUG] overall_conf_avg keys: {list(overall_conf_avg.keys())}")
    print(f"[DEBUG] step_fracs length: {len(step_fracs)}")
    
    # Per-dataset confidence curves
    for dataset_name, curves_dict in dataset_conf_avg.items():
        if not curves_dict:
            continue
        dpath = root / dataset_name
        fig_dir = dpath / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_confidence_curves(curves_dict, step_fracs, fig_dir / "avg_confidence_curves.png",
                              f"GT Confidence Curves ({dataset_name.upper()}, resnet18)")
    
    # Overall confidence curves
    step_fracs_overall = overall_conf_avg.pop("step_fracs", [])
    # Check if there are methods with data (not just step_fracs)
    methods_with_data = [k for k in overall_conf_avg.keys() if k != "step_fracs"]
    
    if methods_with_data and step_fracs_overall:
        print(f"[DEBUG] Plotting overall curves with methods: {methods_with_data}")
        plot_confidence_curves(overall_conf_avg, step_fracs_overall,
                              summary_dir / "figures" / "overall_confidence_curves.png",
                              "GT Confidence Curves (Overall, all datasets, resnet18)")
    else:
        print(f"[INFO] No confidence curves data available for overall plot")
        print(f"[INFO] Methods found: {methods_with_data}, step_fracs: {len(step_fracs_overall)}")

    print("\n✓ Aggregation and comprehensive plotting complete")


if __name__ == "__main__":
    main()
