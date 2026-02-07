#!/bin/bash

# Grid ISIC Bulk Certification (resnet18, multi-head)
# Runs inside docker container with conda environment activated
# Accepts arguments from HTCondor

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Grid ISIC Bulk Certification (resnet18)"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Python Binary: $CONDA_PYTHON_BINARY_PATH"
echo "Arguments: $@"
echo "=========================================="

$CONDA_PYTHON_BINARY_PATH run_bulk_certify_grid_isic.py \
  --project_root "$PROJECT_ROOT" \
  --python "$CONDA_PYTHON_BINARY_PATH" \
  "$@"

echo "=========================================="
echo "Grid ISIC Bulk Certification COMPLETE"
echo "=========================================="
