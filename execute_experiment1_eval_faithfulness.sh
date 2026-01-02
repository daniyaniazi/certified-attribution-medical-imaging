#!/bin/bash
# Execute faithfulness evaluation for Experiment 1
# Loads certification results and computes faithfulness metrics
# Run from repo root: bash execute_experiment1_eval_faithfulness.sh

cd "$(dirname "$0")" || exit 1

echo "=========================================="
echo "Experiment 1: Faithfulness Evaluation"
echo "=========================================="

python experiment1_eval_faithfulness.py \
  --certifications_root outputs/certifications \
  --output_root outputs/experiment1/faithfulness \
  --checkpoint_root outputs/checkpoints \
  --datasets brain_mri chestxray fundus isic \
  --models resnet18 densenet121 mobilenet_v2 efficientnet_b1 \
  --device cuda \
  --deletion_steps 5

echo "Done!"
