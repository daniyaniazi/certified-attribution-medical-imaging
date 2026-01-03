#!/usr/bin/env python
"""Aggregate and plot robustness results for Experiment 2 (bulk certification).

Inputs:
  outputs/eval/experiment2/<dataset>/resnet18/robustness_results.json

Outputs (written to outputs/eval/experiment2/summary/):
  - summary.json : structured aggregates
  - summary.csv  : tabular per-dataset aggregates
  - README.txt   : quick description

Aggregations:
  * per dataset (single resnet18 model)
  * overall: weighted mean over all datasets
  * per attribution method: global mean across all datasets
  * per attribution method by dataset
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Colorblind-friendly palette (Okabe-Ito inspired)
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
    """Aggregate across methods and K for a single dataset's robustness_results.json."""
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


def plot_stacked_avg(aggregated_results, output_path: Path, title_suffix: str):
    """Plot stacked bar chart for aggregated robustness results."""
    methods = sorted(aggregated_results.keys())
    k_values = sorted(next(iter(aggregated_results.values())).keys(), reverse=True) if methods else []
    
    colors = [METHOD_COLORS.get(m, f'C{i}') for i, m in enumerate(methods)]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(k_values))
    width = 0.12
    
    for method_idx, method in enumerate(methods):
        certified_1_list = []
        certified_0_list = []
        abstain_list = []
        
        for k_percent in k_values:
            metrics = aggregated_results[method].get(k_percent, {})
            c1 = metrics.get("mean_pct_certified_1", 0.0)
            c0 = metrics.get("mean_pct_certified_0", 0.0)
            ab = 100.0 - c1 - c0
            certified_1_list.append(c1)
            certified_0_list.append(c0)
            abstain_list.append(max(0, ab))
        
        offset = (method_idx - len(methods) / 2 + 0.5) * width
        
        ax.bar(x + offset, certified_1_list, width, label=method if method_idx == 0 else '',
              color=colors[method_idx], alpha=1.0, edgecolor='black', linewidth=0.5)
        ax.bar(x + offset, certified_0_list, width, bottom=certified_1_list,
              color=colors[method_idx], alpha=0.65, edgecolor='black', linewidth=0.5)
        ax.bar(x + offset, abstain_list, width,
              bottom=np.array(certified_1_list) + np.array(certified_0_list),
              color='white', alpha=1.0, edgecolor='#CCCCCC', linewidth=0.5)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[i], edgecolor='black', label=m)
        for i, m in enumerate(methods)
    ]
    legend_elements.extend([
        mpatches.Patch(facecolor='darkgray', label='Certified 1', alpha=1.0),
        mpatches.Patch(facecolor='gray', label='Certified 0', alpha=0.65),
        mpatches.Patch(facecolor='white', edgecolor='#CCCCCC', label='Abstain'),
    ])
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}%' for k in k_values], fontsize=11)
    ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title(f'Robustness Certification {title_suffix}', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Stacked avg plot: {output_path}")


def aggregate_by_method(root: Path):
    """Aggregate pct_certified per attribution method across all datasets."""
    method_vals = defaultdict(list)
    for dpath in root.iterdir():
        if not dpath.is_dir() or dpath.name == "summary":
            continue
        for mdir in dpath.iterdir():
            if not mdir.is_dir():
                continue
            rr_path = mdir / "robustness_results.json"
            if not rr_path.exists():
                continue
            rr = load_json(rr_path)
            for method_name, kdict in rr.items():
                for _k, metrics in kdict.items():
                    n = metrics.get("num_images", 0)
                    v = metrics.get("pct_certified", 0.0)
                    method_vals[method_name].append((v, n))
    
    method_aggs = {}
    for method, vals in method_vals.items():
        mean_cert, total_n = weighted_mean(vals)
        method_aggs[method] = {
            "mean_pct_certified": mean_cert,
            "total_images": total_n,
        }
    return method_aggs


def plot_global_method_stacked(method_aggs: dict, out_dir: Path):
    """Plot stacked bar of avg pct_certified per attribution method."""
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(method_aggs.keys())
    vals = [method_aggs[m]["mean_pct_certified"] for m in methods]
    
    colors = [METHOD_COLORS.get(m, f'C{i}') for i, m in enumerate(methods)]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, vals, color=colors, edgecolor="black", linewidth=1.2)
    plt.ylabel("Mean % certified", fontsize=12, fontweight="bold")
    plt.ylim(0, 100)
    plt.title("Overall attribution method robustness (all datasets)")
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_path = out_dir / "global_method_robustness.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  ✓ Global method bar written to {out_path}")


def aggregate_by_dataset_and_method(root: Path):
    """Aggregate pct_certified per method for each dataset."""
    dataset_method = defaultdict(lambda: defaultdict(list))
    for dpath in root.iterdir():
        if not dpath.is_dir() or dpath.name == "summary":
            continue
        dataset_name = dpath.name
        for mdir in dpath.iterdir():
            if not mdir.is_dir():
                continue
            rr_path = mdir / "robustness_results.json"
            if not rr_path.exists():
                continue
            rr = load_json(rr_path)
            for method_name, kdict in rr.items():
                for _k, metrics in kdict.items():
                    n = metrics.get("num_images", 0)
                    v = metrics.get("pct_certified", 0.0)
                    dataset_method[dataset_name][method_name].append((v, n))
    
    dataset_method_aggs = {}
    for dataset_name, method_vals in dataset_method.items():
        dataset_method_aggs[dataset_name] = {}
        for method, vals in method_vals.items():
            mean_cert, total_n = weighted_mean(vals)
            dataset_method_aggs[dataset_name][method] = {
                "mean_pct_certified": mean_cert,
                "total_images": total_n,
            }
    return dataset_method_aggs


def plot_method_by_dataset(dataset_method_aggs: dict, out_dir: Path):
    """Single figure: grouped bars per dataset showing mean %certified per attribution method."""
    if not dataset_method_aggs:
        print("[WARN] No dataset/method aggregates found; skipping method-by-dataset plot")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(dataset_method_aggs.keys())
    methods = sorted({m for d in datasets for m in dataset_method_aggs[d].keys()})
    x = np.arange(len(datasets))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))

    for idx, method in enumerate(methods):
        vals = []
        for d in datasets:
            vals.append(dataset_method_aggs[d].get(method, {}).get("mean_pct_certified", 0.0))
        offset = (idx - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=method, color=METHOD_COLORS.get(method, f"C{idx}"), edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=11)
    ax.set_ylabel("Mean % certified", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title("Attribution robustness by dataset (mean % certified)", fontsize=13, fontweight="bold")
    ax.legend(title="Attribution method", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    plt.tight_layout()
    out_path = out_dir / "method_by_dataset.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Method-by-dataset comparison written to {out_path}")


def plot_summary_bar(summary_json: Path, out_dir: Path):
    """Plot summary bar of mean % certified per dataset."""
    data = load_json(summary_json)
    per_dataset = data.get("per_dataset", {})
    labels = list(per_dataset.keys())
    vals = [per_dataset[k]["mean_pct_certified"] for k in labels]

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, vals, color="#4C8BF5", edgecolor="black")
    plt.ylabel("Mean % certified", fontsize=12, fontweight="bold")
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
    parser = argparse.ArgumentParser(description="Aggregate and plot robustness summaries for Experiment 2")
    parser.add_argument(
        "--robustness_root",
        default="outputs/eval/experiment2",
        help="Root directory containing per-dataset robustness results",
    )
    args = parser.parse_args()

    root = Path(args.robustness_root)
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
        model_dirs = [m for m in dpath.iterdir() if m.is_dir() and (m / "robustness_results.json").exists()]
        if not model_dirs:
            print(f"[WARN] No robustness results for {dataset_name}")
            continue

        # Load and aggregate (usually just one model per dataset)
        for mdir in model_dirs:
            model_name = mdir.name
            rr = load_json(mdir / "robustness_results.json")
            agg = aggregate_dataset(rr)
            summary["per_dataset"][dataset_name] = agg
            overall_vals.append((agg["mean_pct_certified"], agg["total_images"]))
            csv_rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "mean_pct_certified": agg["mean_pct_certified"],
                "total_images": agg["total_images"],
            })

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
    readme = """Experiment 2 Robustness Summary

This directory contains aggregated robustness metrics from outputs/eval/experiment2 (bulk certification).

Fields:
- mean_pct_certified: weighted average of pct_certified across methods and K (weights = num_images per entry)
- total_images: total number of images summed across methods/K for weighting

Files:
- summary.json: per-dataset aggregates and overall mean
- summary.csv: tabular per-dataset mean_pct_certified, total_images
"""
    with open(summary_dir / "README.txt", "w") as f:
        f.write(readme)

    print(f"✓ Summary written to {summary_dir}")

    # Generate plots
    print("\n" + "="*60)
    print("Generating plots")
    print("="*60)

    # Global method robustness
    method_aggs = aggregate_by_method(root)
    plot_global_method_stacked(method_aggs, summary_dir / "figures")

    # Method by dataset
    dataset_method_aggs = aggregate_by_dataset_and_method(root)
    plot_method_by_dataset(dataset_method_aggs, summary_dir / "figures")

    # Summary bar
    summary_json = summary_dir / "summary.json"
    if summary_json.exists():
        plot_summary_bar(summary_json, summary_dir / "figures")

    print("\n✓ Experiment 2 robustness aggregation complete")


if __name__ == "__main__":
    main()
