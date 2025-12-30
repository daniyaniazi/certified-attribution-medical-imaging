#!/bin/bash

# ChestXray Bulk Certification Job for resnet18
# Runs inside docker container with conda environment activated
# Accepts arguments from HTCondor

cd "$PROJECT_ROOT"

echo "=========================================="
echo "ChestXray Bulk Certification (resnet18)"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Python Binary: $CONDA_PYTHON_BINARY_PATH"
echo "Arguments: $@"
echo "=========================================="

$CONDA_PYTHON_BINARY_PATH run_bulk_certify_chestxray.py \
  --project_root "$PROJECT_ROOT" \
  --python "$CONDA_PYTHON_BINARY_PATH" \
  "$@"

echo "=========================================="
echo "ChestXray Bulk Certification COMPLETE"
echo "=========================================="
