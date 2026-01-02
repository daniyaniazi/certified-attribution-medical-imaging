Faithfulness summary

This directory contains aggregated faithfulness metrics from outputs/experiment1/faithfulness.

Fields:
- mean_auc: weighted average of mean_auc across methods and K (weights = num_images per entry)
- total_images: total number of images summed across methods/K for weighting

Files:
- summary.json: per-dataset/per-model aggregates and overall mean
- summary.csv: tabular per-dataset/per-model mean_auc, total_images
