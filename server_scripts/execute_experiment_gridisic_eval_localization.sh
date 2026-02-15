#!/usr/bin/env bash
set -euo pipefail

# Grid ISIC Localization Evaluation execution script for HTCondor
# Usage: Executed by HTCondor via experiment_gridisic_eval_localization.sub

PROJECT_ROOT="${PROJECT_ROOT:-/home/exml_team007/certified-attribution-medical-imaging}"
CONDA_PYTHON_BINARY_PATH="${CONDA_PYTHON_BINARY_PATH:-/home/exml_team007/miniconda3/envs/certified-attribution-medical-imaging/bin/python}"
PYTHON_SCRIPT="server_scripts/experiment_gridisic_eval_localization.py"

cd "$PROJECT_ROOT"
echo "Running Grid ISIC Localization Evaluation: $PYTHON_SCRIPT with Python: $CONDA_PYTHON_BINARY_PATH" >&2
"$CONDA_PYTHON_BINARY_PATH" "$PYTHON_SCRIPT" \
    --cert_results outputs/bulk_certifcation/grid_4/isic/resnet18/results_20260113_013345.pkl \
    --grid_metadata data/raw/grid/isic/val/metadata.json \
    --output_dir outputs/eval/grid/isic \
    --model_name resnet18 \
    --save_per_k_plots "$@"


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
