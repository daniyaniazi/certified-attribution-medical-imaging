#!/bin/bash
set -euo pipefail

# Bulk certification runner for all datasets using resnet18
# Outputs go to outputs/bulk_certification/<dataset>/resnet18/
# Num images to certify: 100 (high-confidence correct ones)

PYTHON_BIN=${PYTHON:-python}
PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$PROJECT_ROOT"

run_cert() {
  local dataset="$1"
  local script="$2"
  local checkpoints="$3"
  local data_root="$4"
  local out_base="outputs/bulk_certification/${dataset}/resnet18"
  local heatmaps="${out_base}/certified_maps"

  echo "===== ${dataset}: resnet18 certification (100 images) ====="
  "$PYTHON_BIN" "$script" \
    --checkpoint_dir "$checkpoints" \
    --data_root "$data_root" \
    --split val \
    --certify_dir "$out_base" \
    --heatmap_dir "$heatmaps" \
    --num_images 100 \
    --num_samples 100 \
    --batch_size 16 \
    --sigma 0.15 \
    --tau 0.75 \
    --alpha 0.001 \
    --k_percents 50 25 5 \
    --save_noisy_samples \
    --panel_examples 3 \
    --models resnet18
}

run_cert "isic" "${PROJECT_ROOT}/certify_isic_server.py" "outputs/checkpoints/isic" "data/raw/isic"
run_cert "chestxray" "${PROJECT_ROOT}/certify_chestxray_server.py" "outputs/checkpoints/chestxray" "data/raw/chestxray"
run_cert "brain_mri" "${PROJECT_ROOT}/certify_brain_mri_server.py" "outputs/checkpoints/brain_mri" "data/raw/brain_mri"
run_cert "fundus" "${PROJECT_ROOT}/certify_fundus_server.py" "outputs/checkpoints/fundus" "data/raw/fundus"

echo "All certifications submitted (resnet18, 100 images each)."
