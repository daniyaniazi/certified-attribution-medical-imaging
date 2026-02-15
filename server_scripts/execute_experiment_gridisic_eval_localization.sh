#!/bin/bash
#
# Execute Grid ISIC Localization Evaluation
#
# This script evaluates the localization performance (GridPG metric)
# of certified attributions on the Grid ISIC dataset.
#
# Prerequisites:
#   1. Grid dataset generated (with metadata.json)
#   2. Bulk certification completed (cert_results.pkl exists)
#
# Outputs:
#   - outputs/eval/grid/isic/localization_results.json
#   - outputs/eval/grid/isic/localization_gridpg.png
#

set -euo pipefail

echo "=========================================="
echo "Grid ISIC Localization Evaluation"
echo "=========================================="
echo "Started at: $(date)"
echo ""

# ============================================================================
# Configuration
# ============================================================================

# Project paths
PYTHON_BIN="${CONDA_PYTHON_BINARY_PATH:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SCRIPT="${PROJECT_ROOT}/server_scripts/experiment_gridisic_eval_localization.py"

# Input paths
CERT_RESULTS="${PROJECT_ROOT}/outputs/bulk_certifcation/grid/isic_2/resnet18/cert_results.pkl"
GRID_METADATA="${PROJECT_ROOT}/data/raw/grid/isic/metadata.json"

# Output paths
OUTPUT_DIR="${PROJECT_ROOT}/outputs/eval/grid/isic"

# Model configuration
MODEL_NAME="resnet18"

echo "Configuration:"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Python: ${PYTHON_BIN}"
echo "  Script: ${SCRIPT}"
echo "  Certification results: ${CERT_RESULTS}"
echo "  Grid metadata: ${GRID_METADATA}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Model: ${MODEL_NAME}"
echo ""

# ============================================================================
# Validation
# ============================================================================

echo "Validating inputs..."

if [ ! -f "${SCRIPT}" ]; then
    echo "❌ Evaluation script not found: ${SCRIPT}"
    exit 1
fi

if [ ! -f "${CERT_RESULTS}" ]; then
    echo "❌ Certification results not found: ${CERT_RESULTS}"
    echo "   Please run bulk certification first:"
    echo "   python server_scripts/run_bulk_certify_grid_isic.py --grid_pt <path_to_grid.pt>"
    exit 1
fi

if [ ! -f "${GRID_METADATA}" ]; then
    echo "❌ Grid metadata not found: ${GRID_METADATA}"
    echo "   Ensure grid dataset was generated with metadata.json"
    exit 1
fi

echo "✓ All inputs validated"
echo ""

# ============================================================================
# Create output directory
# ============================================================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/outputs/logs"

# ============================================================================
# Run Evaluation
# ============================================================================

echo "=========================================="
echo "Running Localization Evaluation"
echo "=========================================="
echo ""

${PYTHON_BIN} "${SCRIPT}" \
    --cert_results "${CERT_RESULTS}" \
    --grid_metadata "${GRID_METADATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --save_per_k_plots

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Localization Evaluation Completed Successfully"
    echo "=========================================="
    echo ""
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "  - Metrics: ${OUTPUT_DIR}/localization_results.json"
    echo "  - Main plot: ${OUTPUT_DIR}/localization_gridpg.png"
    echo "  - Per-K plots: ${OUTPUT_DIR}/localization_k*.png"
    echo ""
    echo "Next steps:"
    echo "  1. Review GridPG scores in localization_results.json"
    echo "  2. Check visualization in localization_gridpg.png"
    echo "  3. Compare methods (GridPG ≥ 0.25 = better than random)"
else
    echo "❌ Localization Evaluation Failed"
    echo "=========================================="
    echo "Exit code: ${EXIT_CODE}"
    echo "Check logs for details"
fi

echo ""
echo "Finished at: $(date)"
echo ""

exit ${EXIT_CODE}
