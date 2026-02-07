#!/bin/bash

# Experiment 2 Faithfulness Evaluation
# Computes faithfulness metrics from bulk certification results.
# Expects PROJECT_ROOT and CONDA_PYTHON_BINARY_PATH in environment.

set -euo pipefail

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Experiment2 Faithfulness Evaluation"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Python Binary: $CONDA_PYTHON_BINARY_PATH"
echo "Arguments: $@"
echo "=========================================="

$CONDA_PYTHON_BINARY_PATH experiment2_eval_faithfulness.py "$@"

echo "=========================================="
echo "Experiment2 Faithfulness Evaluation COMPLETE"
echo "=========================================="
