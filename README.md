# Pixel-Level Certified Explanations for Medical Imaging

This project studies pixel-level certified attributions for medical imaging classifiers using randomized smoothing and top-K sparsification. Following [Anani et al. 2025](https://arxiv.org/abs/2506.15499) and grid-based evaluation from [Rao et al. 2022](https://arxiv.org/abs/2205.10435), the repository applies the certification framework to four medical imaging datasets and includes synthetic grid-based evaluation with known ground-truth regions.

![Grid Certification Example](outputs/eval/grid/isic/localization_gridpg.png)
_Example: Grid-based localization (GridPG) scores across methods and K values._

## Overview

Attribution methods explain model predictions by highlighting influential pixels, but [Ghorbani et al. 2019](https://arxiv.org/abs/1810.03292) showed that these explanations can be sensitive to small perturbations even when the prediction remains unchanged. This repository examines that issue in a medical imaging setting.

This project:

- **Certifies** pixel-level attributions across four medical datasets using randomized smoothing
- **Evaluates** five attribution methods (Integrated Gradients, Grad-CAM, LRP, RISE, Occlusion)
- **Tests** localization using synthetic 2x2 grids where the correct region is known by construction

![Attribution Methods Comparison](outputs/attributions_viz/brain_mri/sample_321_all_models.png)
_Attribution visualizations for Brain MRI across five methods: Integrated Gradients, Grad-CAM, RISE, Occlusion, and LRP._

## Datasets

Four medical imaging datasets spanning different modalities and diagnostic tasks:

| Dataset          | Modality           | Task                         | Classes |
| ---------------- | ------------------ | ---------------------------- | ------- |
| ChestX-ray14     | X-ray (grayscale)  | Pneumonia detection          | 2       |
| Brain MRI Tumor  | MRI (grayscale)    | Tumor classification         | 4       |
| APTOS 2019       | Fundus (color)     | Diabetic retinopathy grading | 5       |
| ISIC Skin Lesion | Dermoscopy (color) | Skin lesion classification   | 7       |

![Dataset Samples](outputs/attributions_viz/chestxray/sample_212_all_models.png)
_Representative sample from ChestX-ray14 dataset (with method outputs)._

## Experiments

### Experiment 1: Cross-Architecture Comparison

**Setup:** 5 images per dataset × 4 architectures (ResNet-18, DenseNet-121, MobileNetV2, EfficientNet-B1)

**Goal:** Understand if model architecture affects certification behavior

**Result:** Architecture doesn't significantly change which attribution methods certify best. Gradient-based methods (IG, Grad-CAM) consistently outperform perturbation-based methods (RISE, Occlusion) by 10-20%.

![Experiment 1 Robustness](outputs/eval/experiment1/robustness/brain_mri/resnet18/figures/robustness_stacked.png)
_Robustness results for Brain MRI (ResNet-18). Gradient methods dominate across models._

### Experiment 2: Large-Scale Evaluation

**Setup:** 100 images per dataset × ResNet-18 only

**Goal:** Scale up to validate Experiment 1 findings with statistical confidence

**Result:** The method ordering observed in Experiment 1 is broadly preserved at larger scale. Brain MRI shows the highest average certified percentage (68%), while APTOS fundus shows the lowest (48%). Certification decreases as sparsity increases from $K=50\%$ to $K=25\%$ to $K=5\%$.

![Experiment 2 Summary](outputs/eval/experiment2/robustness/summary/figures/summary_mean_pct_certified.png)
_Average pixel certification rates across datasets. Brain MRI (grayscale) significantly more robust than color datasets._

### Grid-Based Localization

**Setup:** Synthetic 2x2 grids from ISIC images with known target cell (top-left)

**Goal:** Test whether certified pixels localize to the correct region when ground truth is known

**Result:** Localization performance is limited. RISE obtains GridPG = 0.0 across settings, Integrated Gradients drops to 0.0 at sparse $K$, and the strongest method in this setting (Grad-CAM) reaches 0.42, only modestly above the random baseline of 0.25.

![Grid Certification Visualization](outputs/bulk_certifcation/isic/resnet18/resnet18_img185_paper_style.png)
_Grid certification-style panel for ISIC (bulk certification output)._

## Key Findings

### Robustness

- **Gradient methods:** Integrated Gradients and Grad-CAM certify more pixels than the perturbation-based methods in most settings
- **Modality effect:** Brain MRI has higher certification rates than the colour datasets considered here
- **Architecture effect:** Method rankings are similar across the evaluated backbones

![Certified Attribution Example](outputs/certifications/brain_mri/resnet18_img1_paper_style.png)
_Certification panel: input image, five attribution heatmaps, and certified labels at $K=50\%$._

### Faithfulness

- **Deletion curves:** Many curves are flat or non-monotonic, which suggests that certified pixels are not always strongly influential for the prediction
- **Method comparison:** IG and Grad-CAM generally show larger confidence drops than RISE and Occlusion
- **Aggregation issue:** Dataset-level averages can hide substantial image-level variability

![Faithfulness Curves](outputs/eval/experiment2/faithfulness/summary/figures/overall_confidence_curves.png)
_Average faithfulness curves. Flat regions indicate certified pixels have minimal impact on predictions._

### Localization (Grid Evaluation)

- **RISE:** GridPG = 0.0 across all $K$ values
- **Integrated Gradients:** GridPG falls to 0.0 at $K=25\%$ and $K=5\%$ despite stronger performance on standard images
- **Best-performing method:** Grad-CAM peaks at 0.42, compared with the random baseline of 0.25
- **Interpretation:** These results suggest that pixel-level certification does not necessarily imply accurate spatial localization

## Critical Assessment

The grid localization results motivate several questions about certified attributions:

1. **Are we certifying noise?** Randomized smoothing may certify pixel patterns that are stable under perturbation but not semantically meaningful
2. **Method limitations:** Attribution methods designed for single objects struggle with multi-lesion grids
3. **Model quality matters:** If the classifier is weak or uses shortcuts, certified attributions inherit those flaws
4. **Faithfulness concerns:** Flat deletion curves suggest many certified pixels don't actually influence predictions

**Conclusion:** Pixel-level certification is theoretically well motivated, but its practical use may depend on stronger models, improved attribution methods, and more reliable localization behaviour before clinical deployment.

## Installation

```bash
conda env create -f environment.yml
conda activate certified-attribution
```

## Streamlit Interface

To present the full project pipeline and results:

- Dataset preparation
- Architecture and attribution pipeline
- Attribution methods and certification
- Experiment 1 (cross-architecture)
- Experiment 2 (large-scale)
- Grid-based validation

Run:

```bash
streamlit run streamlit_app.py
```

## Project Structure

```
certified-attribution-medical-imaging/
├── src/
│   ├── datasets/          # Medical dataset loaders + grid generation
│   ├── models/            # CNN architectures and multi-head wrapper
│   ├── xai/               # Attribution methods (IG, Grad-CAM, RISE, Occlusion, LRP)
│   └── certify/           # Certification framework with randomized smoothing
├── outputs/
│   ├── certifications/    # Certified attribution visualizations
│   ├── eval/              # Experiment results and metrics
│   └── checkpoints/       # Trained model weights
├── report/                # LaTeX project report with full analysis
└── data/raw/              # Medical imaging datasets

```

## References

- [Anani et al. 2025] "Pixel-level Certified Explanations via Randomized Smoothing" - [arXiv:2506.15499](https://arxiv.org/abs/2506.15499)
- [Rao et al. 2022] "Towards Better Understanding Attribution Methods" - [arXiv:2205.10435](https://arxiv.org/abs/2205.10435)
- [Ghorbani et al. 2019] "Interpretation of Neural Networks is Fragile" - [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)

---

**Technical detail and analysis available in `report/project_report.pdf`**
