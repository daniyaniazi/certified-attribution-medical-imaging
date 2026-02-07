#!/bin/bash
set -euo pipefail

# Navigate to project root
cd "${PROJECT_ROOT}"

# Execute the ChestXray certification script
"${CONDA_PYTHON_BINARY_PATH}" "${PYTHON_SCRIPT}" "$@"
