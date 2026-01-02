Robustness summary

This directory contains aggregated robustness metrics from outputs/eval/experiment1/robustness.

Fields:
- mean_pct_certified: weighted average of pct_certified across methods and K (weights = num_images per entry)
- total_images: total number of images summed across methods/K for weighting

Files:
- summary.json: per-dataset/per-model aggregates and overall mean
- summary.csv: tabular per-dataset/per-model mean_pct_certified, total_images
