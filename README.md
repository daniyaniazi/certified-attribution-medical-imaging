"""
README: Certified Pixel Attribution for Medical Imaging

This project implements certified pixel attribution on medical images.
Based on research paper approach using randomized smoothing + sparsification.

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare data:

   - CheXpert: Download from https://stanfordmlgroup.github.io/competitions/chexpert/
   - ISIC: Download from https://www.isic-archive.com/
   - APTOS: Download from Kaggle competition

3. Run training:

   ```bash
   python src/experiments/run_train.py --config configs/chexpert.yaml --model resnet18
   ```

4. Generate attributions:

   ```bash
   python src/experiments/run_attribution.py --checkpoint outputs/checkpoints/best_model.pt --method integrated_gradients
   ```

5. Certify attributions:

   ```bash
   python src/experiments/run_certify.py --checkpoint outputs/checkpoints/best_model.pt --sigma 0.15 --num_samples 100
   ```

6. Evaluate:
   ```bash
   python src/experiments/run_eval.py --checkpoint outputs/checkpoints/best_model.pt
   ```

## Project Structure

```
certified-attribution-medical-imaging/
├── README.md
├── requirements.txt
├── configs/
│   ├── defaults.py          # Default hyperparameters
│   ├── chexpert.yaml        # CheXpert config
│   ├── isic.yaml            # ISIC config
│   └── aptos.yaml           # APTOS config
├── data/
│   ├── raw/                 # Download datasets here
│   └── processed/           # Processed datasets
├── src/
│   ├── datasets/
│   │   ├── base.py          # Base dataset class
│   │   ├── chexpert.py      # CheXpert loader
│   │   ├── isic.py          # ISIC loader
│   │   └── aptos.py         # APTOS loader
│   ├── models/
│   │   └── factory.py       # Model factory (ResNet, DenseNet, etc.)
│   ├── train/
│   │   ├── train_one.py     # Training loop
│   │   ├── evaluate.py      # Evaluation functions
│   │   └── metrics.py       # Metric computation
│   ├── xai/
│   │   ├── attribution.py   # Attribution methods (IG, Grad-CAM, RISE, Occlusion)
│   │   └── utils.py         # XAI utilities
│   ├── certify/
│   │   ├── sparsify.py      # Sparsification
│   │   ├── smoothing.py     # Randomized smoothing
│   │   └── evaluate.py      # Certification evaluation
│   ├── experiments/
│   │   ├── run_train.py     # Training script
│   │   ├── run_attribution.py  # Attribution generation
│   │   ├── run_certify.py   # Certification script
│   │   └── run_eval.py      # Evaluation script
│   └── utils/
│       ├── seed.py          # Reproducibility
│       ├── io.py            # I/O utilities
│       └── viz.py           # Visualization
└── outputs/
    ├── checkpoints/         # Model checkpoints
    ├── attributions_raw/    # Non-certified attributions
    ├── attributions_certified/ # Certified attributions
    └── reports/             # Evaluation reports

```

## Core Concepts

### 1. Attribution Methods

- **Integrated Gradients (IG)**: Gradient-based method integrating from baseline
- **Grad-CAM**: Class activation mapping using gradients
- **RISE**: Randomized input sampling for explanation
- **Occlusion**: Perturbation-based method

### 2. Certification Pipeline

1. **Sparsification**: Keep only top-K% pixels from attribution
2. **Randomized Smoothing**: Add Gaussian noise N(0, σ²I) and recompute attributions
3. **Aggregation**: Majority voting per pixel across noisy samples
4. **Threshold**: Certify pixels where confidence > τ

### 3. Evaluation Metrics

- **Faithfulness (Deletion AUC)**: Model confidence drop when deleting important pixels
- **Localization (IoU)**: Overlap with ground truth masks (where available)
- **Robustness (%Certified)**: Percentage of pixels with certification

## Hyperparameters

Key certification hyperparameters (from paper):

- `sigma`: Gaussian noise std (default: 0.15)
- `tau`: Certification threshold (default: 0.75)
- `num_samples`: Number of smoothing samples (default: 100)
- `K`: Sparsification percentiles (default: [50, 30, 10])
- `alpha`: Significance level (default: 0.001)

## Example: End-to-End Workflow

```python
from src.models.factory import get_model
from src.datasets.chexpert import CheXpertDataset
from src.train.train_one import Trainer
from src.xai.attribution import IntegratedGradients
from src.certify.smoothing import RandomizedSmoothingAttributor
from src.certify.sparsify import sparsify_topk
from src.certify.evaluate import CertificationEvaluator

# 1. Load model
model, config = get_model('resnet18', num_classes=2)

# 2. Load data
dataset = CheXpertDataset(split='test')

# 3. Create attribution method
ig = IntegratedGradients(model)

# 4. Create smoother
smoother = RandomizedSmoothingAttributor(model, ig.attribute)

# 5. Certify single image
image = dataset[0]['image']  # [C, H, W]
attr = ig.attribute(image, target_class=1)
sparse_attr = sparsify_topk(attr, k_percent=30)
certified, votes, pct_cert = smoother.certify(
    image, sparse_attr, target_class=1,
    sigma=0.15, num_samples=100, tau=0.75
)

# 6. Evaluate
evaluator = CertificationEvaluator()
metrics = evaluator.evaluate_certified(certified, votes, num_samples=100)
print(f"Certified: {metrics['pct_certified']:.1f}%")
```

## Citation

If you use this code, please cite the original paper:

```
@article{...
}
```

## License

MIT License
"""
