#!/bin/bash

# Experiment 2 Robustness Evaluation (Bulk Certification)
# Aggregates bulk certification pickles into robustness metrics/figures per dataset/model.
# Expects PROJECT_ROOT and CONDA_PYTHON_BINARY_PATH in environment.

set -euo pipefail

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Experiment2 Robustness Evaluation"
echo "Inputs: outputs/bulk_certifcation/<dataset>/resnet18/"
echo "Outputs: outputs/eval/experiment2/<dataset>/resnet18"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Python Binary: $CONDA_PYTHON_BINARY_PATH"
echo "Arguments: $@"
echo "=========================================="

$CONDA_PYTHON_BINARY_PATH experiment2_eval_robustness.py "$@"

echo "=========================================="
echo "Experiment2 Robustness Evaluation COMPLETE"
echo "=========================================="
