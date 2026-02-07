#!/usr/bin/env python
"""Aggregate and plot faithfulness results across datasets/models for Experiment 1.

Inputs:
  outputs/eval/experiment1/faithfulness/<dataset>/<model>/faithfulness_results.json

Outputs (written to outputs/eval/experiment1/faithfulness/summary/):
  - summary.json : structured aggregates
  - summary.csv  : tabular per-dataset/per-model aggregates
  - README.txt   : quick description and metric definitions
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


def aggregate_model(result_dict):
    """Aggregate across methods and K for a single model's faithfulness_results.json."""
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


def aggregate_dataset(model_aggs):
    """Aggregate across models for a dataset."""
    vals = [(m["mean_auc"], m["total_images"]) for m in model_aggs]
    mean_auc, total_n = weighted_mean(vals)
    return {
        "mean_auc": mean_auc,
        "total_images": total_n,
    }


def aggregate_across_models(model_results_dict):
    """Aggregate faithfulness results across models, preserving method/K structure.
    
    Args:
        model_results_dict: {model_name: {method: {k: metrics}}}
    
    Returns:
        {method: {k: {mean metrics}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for model_name, methods_dict in model_results_dict.items():
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


def plot_dataset_faithfulness(dataset_name: str, dataset_dir: Path, checkpoint_root: Path, device: str = "cpu"):
    """Plot per-model and dataset-averaged faithfulness for a dataset."""
    dataset_results = {}
    model_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and (d / "faithfulness_results.json").exists()]
    if not model_dirs:
        print(f"[WARN] No model faithfulness files under {dataset_dir}")
        return

    for mdir in model_dirs:
        model_name = mdir.name
        fa_path = mdir / "faithfulness_results.json"
        if fa_path.exists():
            fa = load_json(fa_path)
            dataset_results[model_name] = fa

    # Plot per-model curves
    for mdir in model_dirs:
        model_name = mdir.name
        if model_name in dataset_results:
            fig_dir = mdir / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            plot_auc_curves(
                dataset_results[model_name],
                fig_dir / "faithfulness_auc.png",
                f"(Dataset: {dataset_name.upper()}, Model: {model_name})"
            )
    
    # Plot dataset-averaged curves
    aggregated = aggregate_across_models(dataset_results)
    fig_dir = dataset_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_auc_curves(
        aggregated,
        fig_dir / "dataset_avg_faithfulness.png",
        f"(Dataset: {dataset_name.upper()}, Avg across all models)"
    )
    
    # Also save to dataset root for easy access
    plot_auc_curves(
        aggregated,
        dataset_dir / "avg_faithfulness.png",
        f"(Dataset: {dataset_name.upper()}, Avg across all models)"
    )
    
    return dataset_results


def aggregate_by_method(root: Path):
    """Aggregate mean_auc per attribution method across all datasets/models."""
    method_vals = defaultdict(list)
    for dpath in root.iterdir():
        if not dpath.is_dir() or dpath.name == "summary":
            continue
        for mdir in dpath.iterdir():
            if not mdir.is_dir():
                continue
            fa_path = mdir / "faithfulness_results.json"
            if not fa_path.exists():
                continue
            fa = load_json(fa_path)
            for method_name, kdict in fa.items():
                for k, metrics in kdict.items():
                    n = metrics.get("num_images", 1)
                    auc = metrics.get("mean_auc", 0.0)
                    method_vals[method_name].append((auc, n))
    
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
    plt.title("Overall attribution method faithfulness (all datasets & models)")
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "global_method_faithfulness.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Global method bar written to {out_path}")


def aggregate_by_dataset_and_method(root: Path):
    """Aggregate mean_auc per method for each dataset across all models."""
    dataset_method = defaultdict(lambda: defaultdict(list))
    for dpath in root.iterdir():
        if not dpath.is_dir() or dpath.name == "summary":
            continue
        dataset_name = dpath.name
        for mdir in dpath.iterdir():
            if not mdir.is_dir():
                continue
            fa_path = mdir / "faithfulness_results.json"
            if not fa_path.exists():
                continue
            fa = load_json(fa_path)
            for method_name, kdict in fa.items():
                for k, metrics in kdict.items():
                    n = metrics.get("num_images", 1)
                    auc = metrics.get("mean_auc", 0.0)
                    dataset_method[dataset_name][method_name].append((auc, n))
    
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
        print("[WARN] No dataset-method aggregates found; skipping method-by-dataset plot")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(dataset_method_aggs.keys())
    methods = sorted({m for d in datasets for m in dataset_method_aggs[d].keys()})
    x = np.arange(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(16, 7))

    for idx, method in enumerate(methods):
        vals = []
        for d in datasets:
            vals.append(dataset_method_aggs[d].get(method, {}).get("mean_auc", 0.0))
        offset = (idx - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=method, color=METHOD_COLORS.get(method, f"C{idx}"), edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=11)
    ax.set_ylabel("Mean AUC", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title("Attribution faithfulness by dataset (mean AUC)", fontsize=13, fontweight="bold")
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
    plt.title("Faithfulness summary (mean AUC per dataset)")
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
        
        for mdir in dpath.iterdir():
            if not mdir.is_dir():
                continue
            model_name = mdir.name
            
            # Load confidence curve data from model's figures directory
            conf_curve_file = mdir / "figures" / "faithfulness_confidence_curves_data.json"
            if not conf_curve_file.exists():
                continue
            
            try:
                conf_data = load_json(conf_curve_file)
                step_fracs = conf_data.get("step_fracs", [])
                found_any = True
                
                # Structure: {"step_fracs": [...], "k_values": {"50": {"methods": {...}}, ...}}
                k_values_dict = conf_data.get("k_values", {})
                for k_str, k_entry in k_values_dict.items():
                    k_val = int(k_str)
                    methods_dict = k_entry.get("methods", {})
                    for method, stats in methods_dict.items():
                        mean_conf = stats.get("mean_conf", [])
                        if mean_conf:
                            dataset_curves[dataset_name][method][k_val].append(mean_conf)
                            overall_curves[method][k_val].append(mean_conf)
            except Exception as e:
                print(f"[WARN] Failed to load confidence curves for {dataset_name}/{model_name}: {e}")
                continue
    
    # Average curves per dataset
    dataset_avg = {}
    for dataset_name, method_dict in dataset_curves.items():
        dataset_avg[dataset_name] = {}
        for method, k_dict in method_dict.items():
            dataset_avg[dataset_name][method] = {}
            for k_val, curves_list in k_dict.items():
                if curves_list:
                    dataset_avg[dataset_name][method][k_val] = np.mean(curves_list, axis=0).tolist()
    
    # Average curves overall
    overall_avg = {"step_fracs": step_fracs} if found_any else {}
    for method, k_dict in overall_curves.items():
        overall_avg[method] = {}
        for k_val, curves_list in k_dict.items():
            if curves_list:
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
    parser = argparse.ArgumentParser(description="Aggregate and plot faithfulness summaries for Experiment 1")
    parser.add_argument(
        "--faithfulness_root",
        default="outputs/eval/experiment1/faithfulness",
        help="Root directory containing per-dataset/model faithfulness results",
    )
    parser.add_argument(
        "--checkpoint_root",
        default="outputs/checkpoints",
        help="Root of checkpoints (dataset subfolders) for reference",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for evaluator instantiation (cpu is fine)",
    )
    args = parser.parse_args()

    root = Path(args.faithfulness_root)
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
        model_dirs = [m for m in dpath.iterdir() if m.is_dir() and (m / "faithfulness_results.json").exists()]
        model_aggs = []
        for mdir in model_dirs:
            model_name = mdir.name
            fa_path = mdir / "faithfulness_results.json"
            fa = load_json(fa_path)
            agg = aggregate_model(fa)
            summary["per_model"][f"{dataset_name}/{model_name}"] = agg
            model_aggs.append(agg)
            overall_vals.append((agg["mean_auc"], agg["total_images"]))
            csv_rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "mean_auc": agg["mean_auc"],
                "total_images": agg["total_images"],
            })
        
        if model_aggs:
            summary["per_dataset"][dataset_name] = aggregate_dataset(model_aggs)

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
    csv_rows.sort(key=lambda r: (r["dataset"], r["model"]))
    csv_path = summary_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "mean_auc", "total_images"])
        writer.writeheader()
        writer.writerows(csv_rows)

    # Save README
    readme = """Faithfulness summary

This directory contains aggregated faithfulness metrics from outputs/eval/experiment1/faithfulness.

Fields:
- mean_auc: weighted average of mean_auc across methods and K (weights = num_images per entry)
- total_images: total number of images summed across methods/K for weighting

Files:
- summary.json: per-dataset/per-model aggregates and overall mean
- summary.csv: tabular per-dataset/per-model mean_auc, total_images
"""
    with open(summary_dir / "README.txt", "w") as f:
        f.write(readme)

    print(f"✓ Summary written to {summary_dir}")

    # Plot per-dataset faithfulness and summary bar
    overall_model_results = {}
    for dpath in datasets:
        dataset_name = dpath.name
        dataset_results = plot_dataset_faithfulness(dataset_name, dpath, checkpoint_root, device=args.device)
        if dataset_results:
            overall_model_results.update({f"{dataset_name}_{m}": res for m, res in dataset_results.items()})

    # Overall averaged AUC curves across all datasets/models
    if overall_model_results:
        print("\nGenerating overall average across all datasets/models")
        all_model_results = {dpath.name: {m.name: load_json(m / "faithfulness_results.json") for m in dpath.iterdir() if m.is_dir() and (m / "faithfulness_results.json").exists()} for dpath in datasets}
        flat_results = {}
        for dataset_name, models in all_model_results.items():
            for model_name, results in models.items():
                flat_results[f"{dataset_name}_{model_name}"] = results
        
        overall_aggregated = aggregate_across_models({f"{dataset_name}_{model_name}": res for dataset_name, models in all_model_results.items() for model_name, res in models.items()})
        plot_auc_curves(
            overall_aggregated,
            summary_dir / "figures" / "overall_avg_faithfulness.png",
            "(Overall: Avg across all datasets & models)"
        )
    
    # Global method faithfulness (per method across all datasets/models)
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
        print(f"[WARN] Missing summary.json at {summary_json}; skipping summary plot")

    # Generate confidence curves (Figure 8 style)
    print("\n" + "="*60)
    print("Generating GT confidence curves (Figure 8 style)")
    print("="*60)
    dataset_conf_avg, overall_conf_avg, step_fracs = aggregate_confidence_curves(root)
    
    # Per-dataset confidence curves
    for dataset_name, curves_dict in dataset_conf_avg.items():
        if not curves_dict:
            continue
        plot_confidence_curves(
            curves_dict,
            step_fracs,
            root / dataset_name / "figures" / "avg_confidence_curves.png",
            f"GT Confidence vs Deletion Steps - {dataset_name.upper()} (Avg across models)"
        )
    
    # Overall confidence curves
    if overall_conf_avg:
        step_fracs_overall = overall_conf_avg.pop("step_fracs", [])
        methods_with_data = [k for k in overall_conf_avg.keys() if k != "step_fracs"]
        if methods_with_data and step_fracs_overall:
            plot_confidence_curves(
                overall_conf_avg,
                step_fracs_overall,
                summary_dir / "figures" / "overall_confidence_curves.png",
                "GT Confidence vs Deletion Steps - Overall (Avg across all datasets & models)"
            )

    print("\n✓ Aggregation and comprehensive plotting complete")


if __name__ == "__main__":
    main()
