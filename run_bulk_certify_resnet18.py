#!/usr/bin/env python
"""Bulk certification runner for resnet18 across all datasets.
Outputs: output/bulk_certifcation/<dataset>/resnet18/
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DATASETS = [
    {
        "name": "isic",
        "script": PROJECT_ROOT / "certify_isic_server.py",
        "checkpoints": PROJECT_ROOT / "outputs" / "checkpoints" / "isic",
        "data_root": PROJECT_ROOT / "data" / "raw" / "isic",
    },
    {
        "name": "chestxray",
        "script": PROJECT_ROOT / "certify_chestxray_server.py",
        "checkpoints": PROJECT_ROOT / "outputs" / "checkpoints" / "chestxray",
        "data_root": PROJECT_ROOT / "data" / "raw" / "chestxray",
    },
    {
        "name": "brain_mri",
        "script": PROJECT_ROOT / "certify_brain_mri_server.py",
        "checkpoints": PROJECT_ROOT / "outputs" / "checkpoints" / "brain_mri",
        "data_root": PROJECT_ROOT / "data" / "raw" / "brain_mri",
    },
    {
        "name": "fundus",
        "script": PROJECT_ROOT / "certify_fundus_server.py",
        "checkpoints": PROJECT_ROOT / "outputs" / "checkpoints" / "fundus",
        "data_root": PROJECT_ROOT / "data" / "raw" / "fundus",
    },
]


def run_cert(dataset: dict):
    name = dataset["name"]
    out_base = PROJECT_ROOT / "output" / "bulk_certifcation" / name / "resnet18"
    heatmap_dir = out_base / "certified_maps"
    out_base.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        str(dataset["script"]),
        "--checkpoint_dir",
        str(dataset["checkpoints"]),
        "--data_root",
        str(dataset["data_root"]),
        "--split",
        "val",
        "--certify_dir",
        str(out_base),
        "--heatmap_dir",
        str(heatmap_dir),
        "--num_images",
        "100",
        "--num_samples",
        "100",
        "--batch_size",
        "16",
        "--sigma",
        "0.15",
        "--tau",
        "0.75",
        "--alpha",
        "0.001",
        "--k_percents",
        "50",
        "25",
        "5",
        "--save_noisy_samples",
        "--panel_examples",
        "3",
        "--models",
        "resnet18",
    ]

    print(f"===== {name}: resnet18 certification (100 images) =====")
    subprocess.run(cmd, check=True)


def main():
    for ds in DATASETS:
        run_cert(ds)
    print("All certifications done (resnet18).")


if __name__ == "__main__":
    main()
