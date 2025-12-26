#!/usr/bin/env bash
set -euo pipefail

# Root of your project checkout on the cluster
PROJECT_ROOT="${PROJECT_ROOT:-/home/exml_team007/certified-attribution-medical-imaging}"

# Conda Python inside your env (adjust env name if different)
CONDA_PYTHON_BINARY_PATH="${CONDA_PYTHON_BINARY_PATH:-/home/exml_team007/miniconda3/envs/certified-attribution-medical-imaging/bin/python}"

# Default script to run (can be overridden by passing args in the .sub file)
PYTHON_SCRIPT="${PYTHON_SCRIPT:-train_isic_server.py}"

cd "$PROJECT_ROOT"
echo "Running $PYTHON_SCRIPT with Python: $CONDA_PYTHON_BINARY_PATH" >&2
"$CONDA_PYTHON_BINARY_PATH" "$PYTHON_SCRIPT" "$@"