#!/usr/bin/env python
"""Bulk attribution runner for resnet18 across all datasets.
Outputs: output/bulk_attrubution/<dataset>/resnet18/
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

DATASETS = [
    {
        "name": "isic",
        "script": PROJECT_ROOT / "attribution_isic_server.py",
        "data_root": PROJECT_ROOT / "data" / "raw" / "isic",
    },
    {
        "name": "chestxray",
        "script": PROJECT_ROOT / "attribution_chestxray_server.py",
        "data_root": PROJECT_ROOT / "data" / "raw" / "chestxray",
    },
    {
        "name": "brain_mri",
        "script": PROJECT_ROOT / "attribution_brain_mri_server.py",
        "data_root": PROJECT_ROOT / "data" / "raw" / "brain_mri",
    },
    {
        "name": "fundus",
        "script": PROJECT_ROOT / "attribution_fundus_server.py",
        "data_root": PROJECT_ROOT / "data" / "raw" / "fundus",
    },
]


def run_attr(dataset: dict):
    name = dataset["name"]
    out_base = PROJECT_ROOT / "output" / "bulk_attrubution" / name / "resnet18"
    log_dir = out_base / "logs"
    out_base.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        str(dataset["script"]),
        "--data_root",
        str(dataset["data_root"]),
        "--output_dir",
        str(out_base),
        "--log_dir",
        str(log_dir),
        "--checkpoint_dir",
        str(PROJECT_ROOT / "outputs" / "checkpoints" / name),
        "--models",
        "resnet18",
        "--num_samples",
        "10",
        "--batch_size",
        "4",
        "--num_workers",
        "4",
        "--methods",
        "IntegratedGradients",
        "GradCAM",
        "RISE",
        "Occlusion",
        "LRP",
    ]

    print(f"===== {name}: resnet18 attribution (10 samples) =====")
    subprocess.run(cmd, check=True)


def main():
    for ds in DATASETS:
        run_attr(ds)
    print("All attributions done (resnet18).")


if __name__ == "__main__":
    main()
