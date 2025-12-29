#!/usr/bin/env bash
set -euo pipefail

# Fundus attribution execution script for HTCondor
# Usage: Executed by HTCondor via attribution_fundus.sub

PROJECT_ROOT="${PROJECT_ROOT:-/home/exml_team007/certified-attribution-medical-imaging}"
CONDA_PYTHON_BINARY_PATH="${CONDA_PYTHON_BINARY_PATH:-/home/exml_team007/miniconda3/envs/certified-attribution-medical-imaging/bin/python}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-attribution_fundus_server.py}"

cd "$PROJECT_ROOT"
echo "Running Fundus Attribution Generation: $PYTHON_SCRIPT with Python: $CONDA_PYTHON_BINARY_PATH" >&2
"$CONDA_PYTHON_BINARY_PATH" "$PYTHON_SCRIPT" "$@"
