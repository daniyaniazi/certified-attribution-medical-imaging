#!/bin/bash
set -euo pipefail

# Bulk attribution runner for all datasets using resnet18
# Outputs go to outputs/bulk_attribution/<dataset>/resnet18/
# Num samples: 10 shuffled validation images (script uses np.random.choice with seed 42)

PYTHON_BIN=${PYTHON:-python}
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_ROOT"

run_attr() {
  local dataset="$1"
  local script="$2"
  local data_root="$3"
  local out_base="outputs/bulk_attribution/${dataset}"

  echo "===== ${dataset}: resnet18 attribution (10 samples) ====="
  "$PYTHON_BIN" "$script" \
    --data_root "$data_root" \
    --output_dir "$out_base" \
    --log_dir "${out_base}/logs" \
    --checkpoint_dir "outputs/checkpoints/${dataset}" \
    --models resnet18 \
    --num_samples 10 \
    --batch_size 4 \
    --num_workers 4 \
    --methods IntegratedGradients GradCAM RISE Occlusion LRP
}

run_attr "isic" "${PROJECT_ROOT}/attribution_isic_server.py" "data/raw/isic"
run_attr "chestxray" "${PROJECT_ROOT}/attribution_chestxray_server.py" "data/raw/chestxray"
run_attr "brain_mri" "${PROJECT_ROOT}/attribution_brain_mri_server.py" "data/raw/brain_mri"
run_attr "fundus" "${PROJECT_ROOT}/attribution_fundus_server.py" "data/raw/fundus"

echo "All attributions generated (resnet18, 10 samples each)."
