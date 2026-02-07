#!/bin/bash
set -euo pipefail

cd "${PROJECT_ROOT}"
"${CONDA_PYTHON_BINARY_PATH}" "${PYTHON_SCRIPT}" "$@"
