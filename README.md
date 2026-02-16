# Certified Attribution for Medical Imaging

Pixel-level certified explanations for medical imaging classifiers using randomized smoothing and top-K sparsification. This project implements the framework from [Anani et al. 2025](https://arxiv.org/abs/2506.15499) and applies it to four diverse medical imaging datasets with comprehensive robustness and faithfulness evaluation.

## Project Overview

This repository provides a complete pipeline for:

1. **Training medical image classifiers** across multiple architectures (ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0/B1, MobileNetV2)
2. **Computing pixel-level attributions** using five methods: Integrated Gradients, Grad-CAM, Layer-wise Relevance Propagation (LRP), RISE, and Occlusion
3. **Certifying attributions** via randomized smoothing with per-pixel robustness guarantees
4. **Evaluating robustness and faithfulness** at scale (Experiment 1: 5 images per dataset; Experiment 2: 100 images per dataset)
5. **Grid-based localization evaluation** using synthetic 2×2 grids with ground-truth target regions

## Key Features

- **Four medical datasets**: ChestX-ray14 (X-ray), Brain MRI (MRI), APTOS 2019 (fundus), ISIC (dermoscopy)
- **Unified certification pipeline**: Standard randomized smoothing ($\sigma=0.15$, $n=100$, $\tau=0.75$) applied consistently across all datasets and methods
- **Grid-based DiFull evaluation**: Synthetic $2\times2$ grids enable localization testing with known ground truth target cells
- **Comprehensive metrics**:
  - Robustness: % pixels receiving certified labels (important/unimportant)
  - Faithfulness: pixel deletion curves measuring importance of certified pixels
  - Localization: Certified GridPG metric measuring alignment with target regions
- **Precomputed results**: Paper-style certification visualizations, robustness/faithfulness plots, and localization scores ready for reporting

## Datasets

| Dataset          | Modality           | Task                         | Train/Val/Test | Image Size | Classes |
| ---------------- | ------------------ | ---------------------------- | -------------- | ---------- | ------- |
| ChestX-ray14     | X-ray (grayscale)  | Binary (pneumonia)           | 80/10/10%      | 224×224    | 2       |
| Brain MRI Tumor  | MRI (grayscale)    | 4-class tumor classification | 80/10/10%      | 224×224    | 4       |
| APTOS 2019       | Fundus (color)     | 5-class diabetic retinopathy | 80/10/10%      | 224×224    | 5       |
| ISIC Skin Lesion | Dermoscopy (color) | 7-class skin lesion          | 80/10/10%      | 224×224    | 7       |

## Installation & Setup

### 1. Environment

```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate certified-attribution-medical-imaging

# Or pip
pip install -r requirements.txt
```

### 2. Data Structure

Place datasets under `data/raw/`:

```
data/raw/
├── chestxray/
│   ├── train/
│   │   ├── images/
│   │   └── labels.json
│   ├── val/
│   └── test/
├── brain_mri/
├── fundus/
├── isic/
└── grid/isic/
    ├── val/
    │   ├── grid.pt
    │   ├── metadata.json
    │   └── labels.json
    └── test/
```

## Training Classifiers

Train backbone models on all datasets:

```bash
# Train ResNet-18 on all datasets
python train_all_datasets.py --model resnet18 --epochs 50 --batch_size 32 --device cuda

# Train multiple models for architecture comparison
python train_all_datasets.py --models resnet18 resnet50 densenet121 efficientnet_b1 mobilenetv2 \
    --epochs 50 --batch_size 32 --data_root data/raw --checkpoint_dir outputs/checkpoints
```

Checkpoints saved to `outputs/checkpoints/<dataset>/<model_name>/`.

## Attribution & Certification

### Experiment 1: Cross-Architecture Comparison (5 images per dataset)

Evaluate 4 architectures on a small sample to understand architecture effects:

```bash
python experiment1_eval_attribution.py \
    --datasets chestxray brain_mri fundus isic \
    --models resnet18 densenet121 mobilenetv2 efficientnet_b1 \
    --num_images 5 \
    --output_dir outputs/eval/experiment1/
```

Outputs:

- Robustness plots: `outputs/eval/experiment1/robustness/<dataset>/robustness_stacked.png`
- Faithfulness curves: `outputs/eval/experiment1/faithfulness/<dataset>/faithfulness_confidence_curves.png`
- Summary statistics: `outputs/eval/experiment1/avg_robustness.png`, `overall_confidence_curves.png`

**Result Summary** (Table 1 in report):

- Average certification across all datasets: 57.31%
- Brain MRI most robust (68%), Fundus least robust (52%)
- Gradient-based methods (IG, Grad-CAM) substantially outperform perturbation-based methods (RISE, Occlusion, LRP)

### Experiment 2: Large-Scale Evaluation with ResNet-18 (100 images per dataset)

Scale up to 100 images per dataset for statistical confidence:

```bash
python experiment2_eval_large_scale.py \
    --datasets chestxray brain_mri fundus isic \
    --model resnet18 \
    --num_images 100 \
    --output_dir outputs/eval/experiment2/
```

Outputs:

- Robustness plots and summary: `outputs/eval/experiment2/robustness/`
- Faithfulness curves and summary: `outputs/eval/experiment2/faithfulness/`
- Per-method comparisons: `outputs/eval/experiment2/robustness/method_by_dataset.png`

**Result Summary** (Table 2 in report):

- Large-scale patterns confirm Experiment 1 findings
- Robustness decreases monotonically with K (50% → 25% → 5%)
- Faithfulness curves show erratic behavior; many exhibit flat regions indicating poor pixel importance

### Grid-Based Localization (ISIC only)

Generate synthetic 2×2 grid images and evaluate localization:

```bash
# Step 1: Generate grids (if not precomputed)
python generate_grid_dataset.py \
    --isic_root data/raw/isic \
    --num_grids 100 \
    --output_dir data/raw/grid/isic/val

# Step 2: Compute bulk certifications on grids
python bulk_certify_grid_isic.py \
    --grid_metadata data/raw/grid/isic/val/metadata.json \
    --checkpoint outputs/checkpoints/isic/resnet18/final_model.pt \
    --output_dir outputs/bulk_certifcation/grid_4/isic/resnet18/ \
    --sigma 0.15 --num_samples 100 --tau 0.75 \
    --k_percents 50 25 5

# Step 3: Evaluate localization (GridPG metric)
python experiment_gridisic_eval_localization.py \
    --cert_results outputs/bulk_certifcation/grid_4/isic/resnet18/results_20260113_013345.pkl \
    --grid_metadata data/raw/grid/isic/val/metadata.json \
    --output_dir outputs/eval/grid/isic/ \
    --model_name resnet18 \
    --save_per_k_plots
```

Outputs:

- Localization metrics: `outputs/eval/grid/isic/localization_results.json`
- Visualization: `outputs/eval/grid/isic/localization_gridpg.png` (grouped bar chart)
- Per-K plots: `outputs/eval/grid/isic/localization_k*.png`

**Result Summary** (Table 3 in report):

- Occlusion: best at K=25% (0.3542)
- GradCAM: best at K=5% (0.4204)
- RISE and IntegratedGradients: catastrophic failure (0.0)
- Most methods barely exceed theoretical random expectation (0.25)
- **Critical finding**: Certified pixels fail to localize to ground-truth target cells, raising questions about certification validity

## Key Findings

### Robustness (Experiment 1 & 2)

- **Brain MRI is most robust**: 68% mean certified pixels (high-contrast, well-localized structures)
- **Fundus is least robust**: 48% mean certified pixels (color complexity, distributed vascular patterns)
- **Gradient-based methods win**: Integrated Gradients and Grad-CAM 10-20% higher certification than RISE/Occlusion
- **Monotonic K-effect**: Certification drops as K increases (harder to maintain large top-K sets under perturbations)

### Faithfulness (Experiment 1 & 2)

- **Erratic curve behavior observed**: Many per-image deletion curves are flat, non-monotonic, or show oscillations
- **Aggregation masks failures**: Smooth average curves result from averaging diverse failure modes
- **IG/Grad-CAM better**: Steeper drops when removing certified pixels (40-60% at 30% deletion)
- **RISE/Occlusion unreliable**: Shallow curves suggest certified pixels are not genuinely important

### Localization (Grid ISIC)

- **Poor localization across all methods**: GridPG scores near or below 0.25 theoretical random expectation
- **RISE complete failure**: 0.0 across all K values
- **IntegratedGradients collapses at sparse K**: 0.0 at K=25% and K=5%
- **Best methods modest**: GradCAM reaches 0.4204 at K=5%, Occlusion 0.3542 at K=25%
- **Excessive abstention**: Certified maps are extremely sparse; most pixels marked "abstain" rather than labeled

### Critical Assessment

The grid localization failure (Table 3) casts doubt on standard dataset certification results. If certified pixels fail to align with ground-truth on grids, are they meaningful on real medical images? Analysis suggests:

- Randomized smoothing may certify **noise** rather than robust structure
- Attribution methods poorly suited to images with multiple competing lesions
- Model training insufficient for reliable attribution estimation
- Faithfulness curves' erratic behavior indicates fundamental method limitations

## Project Structure

```
certified-attribution-medical-imaging/
├── src/
│   ├── datasets/          # Dataset loaders for 4 medical datasets + grid generation
│   ├── models/            # CNN architectures and multi-head wrapper for grids
│   ├── train/             # Training pipelines with early stopping, checkpointing
│   ├── certify/           # Certification framework: randomized smoothing, top-K sparsification
│   │   ├── base.py        # BaseEvaluator: model loading, device management
│   │   ├── eval/          # Evaluator classes: robustness, faithfulness, localization
│   │   │   ├── robustness.py
│   │   │   ├── faithfulness.py
│   │   │   └── localization.py  # GridPG metric implementation
│   │   └── certification.py  # Randomized smoothing & top-K pipeline
│   └── xai/               # Attribution methods: IG, Grad-CAM, RISE, Occlusion, LRP
├── server_scripts/        # HTCondor submission scripts for bulk evaluation
│   ├── attribution_*.py    # Per-dataset attribution computation
│   ├── certify_*.py        # Per-dataset certification
│   ├── bulk_certify_*.py   # Large-scale certification
│   ├── experiment1_eval_*.py
│   ├── experiment2_eval_*.py
│   ├── experiment_gridisic_eval_localization.py
│   └── *.sub              # HTCondor submission files
├── data/
│   ├── raw/               # Original datasets
│   └── processed/         # Preprocessed datasets, grids
├── outputs/
│   ├── checkpoints/       # Trained model weights
│   ├── attributions_viz/  # Attribution visualizations
│   ├── certifications/    # Certified attribution panels (paper-style)
│   ├── bulk_certifcation/ # Bulk certification results (pickle format)
│   └── eval/
│       ├── experiment1/   # Exp 1 results: robustness, faithfulness
│       ├── experiment2/   # Exp 2 results: large-scale evaluation
│       └── grid/isic/     # Grid localization metrics and plots
├── report/                # LaTeX project report
│   ├── project_report.tex # Main report with results, tables, figures
│   └── *.bib, *.sty      # Bibliography and style files
└── notebooks/             # Jupyter notebooks for exploration
```

## Configuration

All experiments use consistent hyperparameters:

- **Noise level**: $\sigma = 0.15$ (standard deviation for Gaussian perturbations)
- **Smoothing samples**: $n = 100$ (noisy samples aggregated per image)
- **Certification threshold**: $\tau = 0.75$ (probability threshold for confident labels)
- **Certified radius**: $R \approx 0.101$ (robust to $\ell_2$ perturbations $\|\delta\|_2 \le R$)
- **Sparsity levels**: $K \in \{5\%, 25\%, 50\%\}$ (top-K pixels certified)

## Results Visualization

### Certification Panels (Paper-style)

Individual images showing (left to right): input, attribution heatmaps for 5 methods, certified labels for K=50%

```
outputs/certifications/<dataset>/resnet18_img<N>_paper_style.png
```

### Robustness Plots

Stacked bar charts showing % certified pixels per attribution method

```
outputs/eval/experiment1|2/robustness/<dataset>/robustness_stacked.png
outputs/eval/experiment1|2/robustness/summary_mean_pct_certified.png  # Aggregated
```

### Faithfulness Curves

Pixel deletion curves showing confidence drop as certified important pixels are removed

```
outputs/eval/experiment1|2/faithfulness/<dataset>/faithfulness_confidence_curves.png
outputs/eval/experiment1|2/faithfulness/overall_confidence_curves.png  # Aggregated
```

### Localization Plots

GridPG scores grouped by K value, showing which methods best localize to target cells

```
outputs/eval/grid/isic/localization_gridpg.png
outputs/eval/grid/isic/localization_k{5,25,50}.png  # Per-K breakdowns
```

## Report & Documentation

Comprehensive project report in `report/project_report.tex`:

- Implementation details for all 5 attribution methods
- Algorithm descriptions (top-K sparsification, randomized smoothing, certification)
- Full experimental design and hyperparameter justification
- Results tables and critical analysis
- Discussion of limitations and future work

Generated PDF: `report/project_report.pdf` (compile with pdfLaTeX)

## References

[Anani et al. 2025] "Pixel-level Certified Explanations via Randomized Smoothing" - [arXiv:2506.15499](https://arxiv.org/abs/2506.15499)

## License

This project is provided for educational and research purposes.
