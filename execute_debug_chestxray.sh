#!/bin/bash

# Debug chestxray certification pickle
# Expects PROJECT_ROOT and CONDA_PYTHON_BINARY_PATH in environment.

set -euo pipefail

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Debug Chestxray Certification Pickle"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Python Binary: $CONDA_PYTHON_BINARY_PATH"
echo "=========================================="

$CONDA_PYTHON_BINARY_PATH debug_chestxray_cert.py

echo "=========================================="
echo "Debug Complete"
echo "=========================================="
