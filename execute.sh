#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/exml_team007/certified-attribution-medical-imaging}"
CONDA_PYTHON_BINARY_PATH="${CONDA_PYTHON_BINARY_PATH:-/home/exml_team007/miniconda3/envs/certified-attribution-medical-imaging/bin/python}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-train_isic_server.py}"

cd "$PROJECT_ROOT"
echo "Running $PYTHON_SCRIPT with Python: $CONDA_PYTHON_BINARY_PATH" >&2
"$CONDA_PYTHON_BINARY_PATH" "$PYTHON_SCRIPT" "$@"
