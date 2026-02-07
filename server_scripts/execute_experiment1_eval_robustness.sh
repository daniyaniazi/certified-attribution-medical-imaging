#!/bin/bash

# Experiment 1 Robustness Evaluation
# Aggregates certification pickles into robustness metrics/figures per dataset/model.
# Expects PROJECT_ROOT and CONDA_PYTHON_BINARY_PATH in environment.

set -euo pipefail

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Experiment1 Robustness Evaluation"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Python Binary: $CONDA_PYTHON_BINARY_PATH"
echo "Arguments: $@"
echo "=========================================="

$CONDA_PYTHON_BINARY_PATH experiment1_eval_robustness.py "$@"

echo "=========================================="
echo "Experiment1 Robustness Evaluation COMPLETE"
echo "=========================================="
