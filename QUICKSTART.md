# Quick Start Guide: Certified Pixel Attribution for Medical Imaging

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Project Structure

```
certified-attribution-medical-imaging/
├── src/                      # Source code
│   ├── datasets/             # Dataset loaders
│   ├── models/               # Model architectures
│   ├── train/                # Training pipeline
│   ├── xai/                  # Attribution methods
│   ├── certify/              # Certification pipeline
│   ├── experiments/          # Runnable scripts
│   └── utils/                # Utilities
├── configs/                  # Configuration files
├── data/                     # Datasets (ignored by git)
├── outputs/                  # Results and checkpoints (ignored by git)
└── example.py                # Working examples
```

---

## Workflow

### Step 1: Prepare Data

Download datasets and organize in `data/raw/`:

**CheXpert:**

```bash
# Download from: https://stanfordmlgroup.github.io/competitions/chexpert/
# Place in: data/raw/chexpert/
#   ├── train.csv
#   ├── valid.csv
#   └── [image files]
```

**ISIC:**

```bash
# Download from: https://www.isic-archive.com/
# Place in: data/raw/isic/
#   ├── train/
#   │   ├── images/
#   │   └── labels.json
#   ├── val/
#   │   ├── images/
#   │   └── labels.json
#   └── test/
#       ├── images/
#       └── labels.json
```

### Step 2: Train Model

```bash
python src/experiments/run_train.py \
    --dataset chexpert \
    --model resnet18 \
    --epochs 50
```

**Options:**

- `--dataset`: `chexpert`, `isic`, `aptos`
- `--model`: `resnet18`, `resnet50`, `densenet121`, `efficientnet_b0`, `efficientnet_b1`
- `--epochs`: number of training epochs
- `--lr`: learning rate (default: 1e-3)

**Output:** `outputs/checkpoints/{dataset}/{model}/best_model.pt`

### Step 3: Generate Attribution Maps

```bash
python src/experiments/run_attribution.py \
    --checkpoint outputs/checkpoints/chexpert/resnet18/best_model.pt \
    --dataset chexpert \
    --model resnet18 \
    --method integrated_gradients \
    --save-viz
```

**Attribution Methods:**

- `integrated_gradients`: Gradient-based integration method
- `gradcam`: Class activation mapping
- `rise`: Randomized input sampling
- `occlusion`: Perturbation-based

**Output:** `outputs/attributions_raw/{dataset}/{model}/{method}/`

### Step 4: Certify Attributions

```bash
python src/experiments/run_certify.py \
    --checkpoint outputs/checkpoints/chexpert/resnet18/best_model.pt \
    --dataset chexpert \
    --model resnet18 \
    --method integrated_gradients \
    --sigma 0.15 \
    --tau 0.75 \
    --num-samples 100 \
    --k-percents 50,30,10
```

**Key Parameters:**

- `--sigma`: Gaussian noise std (default: 0.15)
- `--tau`: Certification threshold (default: 0.75)
- `--num-samples`: Number of smoothing samples (default: 100)
- `--k-percents`: Sparsification percentiles

**Output:** `outputs/attributions_certified/{dataset}/{model}/{method}/...`

### Step 5: Evaluate

```bash
python src/experiments/run_eval.py \
    --checkpoint outputs/checkpoints/chexpert/resnet18/best_model.pt \
    --dataset chexpert \
    --model resnet18 \
    --method integrated_gradients
```

**Output:** `outputs/reports/eval_{dataset}_{model}_{method}.json`

---

## Python API Usage

### Train a Model

```python
from src.models.factory import get_model
from src.datasets.chexpert import CheXpertDataset
from src.train.train_one import Trainer
from torch.utils.data import DataLoader

# Load dataset
train_data = CheXpertDataset(split='train')
train_loader = DataLoader(train_data, batch_size=32)

# Create model
model, config = get_model('resnet18', num_classes=2)

# Train
trainer = Trainer(model, train_loader, val_loader, device='cuda')
trainer.fit(epochs=50)
```

### Generate Attribution

```python
from src.xai.attribution import IntegratedGradients
import torch

# Create method
ig = IntegratedGradients(model, device='cuda')

# Compute attribution
image = torch.randn(1, 3, 224, 224)
attribution = ig.attribute(image, target_class=1, num_steps=50)
```

### Certify Attribution

```python
from src.certify.smoothing import RandomizedSmoothingAttributor
from src.certify.sparsify import sparsify_topk

# Create smoother
smoother = RandomizedSmoothingAttributor(model, ig.attribute, device='cuda')

# Sparsify
sparse_attr = sparsify_topk(attribution, k_percent=30)

# Certify
certified, votes, pct_certified = smoother.certify(
    image,
    sparse_attr,
    target_class=1,
    sigma=0.15,
    num_samples=100,
    tau=0.75
)

print(f"Certified: {pct_certified:.1f}%")
```

---

## Expected Directory Structure After Running

```
outputs/
├── checkpoints/
│   └── chexpert/
│       └── resnet18/
│           ├── best_model.pt          # Best trained model
│           └── history.json            # Training history
│
├── attributions_raw/
│   └── chexpert/
│       └── resnet18/
│           └── integrated_gradients/
│               ├── metadata.json
│               ├── 0_attr.npy          # Attribution maps
│               ├── 0_viz.png           # Visualizations
│               └── ...
│
├── attributions_certified/
│   └── chexpert/
│       └── resnet18/
│           └── integrated_gradients/
│               └── sigma0.15_tau0.75_n100/
│                   ├── results.json    # Certification results
│                   ├── 0_attr_base.npy
│                   ├── 0_certified_k50.npy
│                   ├── 0_certified_k30.npy
│                   ├── 0_certified_k10.npy
│                   ├── 0_certified_k50.png
│                   └── ...
│
└── reports/
    └── eval_chexpert_resnet18_integrated_gradients.json
```

---

## Common Issues & Solutions

### 1. "Dataset not found"

**Solution:** Download and place datasets in `data/raw/` according to the structure above.

### 2. GPU memory error

**Solution:** Reduce batch size or num_samples:

```bash
python run_certify.py --batch-size 4 --num-samples 50
```

### 3. CUDA not available

**Solution:** Ensure PyTorch is installed with CUDA support:

```bash
pip install torch torchvision -f https://download.pytorch.org/whl/cu118/torch_stable.html
```

### 4. Missing imports

**Solution:** Make sure you're running from the project root directory and have installed all requirements.

---

## Configuration

All hyperparameters are defined in `configs/defaults.py`. Key settings:

```python
CERTIFICATION_CONFIG = {
    'sigma': 0.15,          # Gaussian noise std
    'tau': 0.75,            # Certification threshold
    'num_samples': 100,     # Smoothing samples
    'batch_size': 16,       # Batch size
    'k_percents': [50, 30, 10]  # Sparsification percentiles
}

TRAINING_CONFIG = {
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'metric_to_track': 'val_auc'
}
```

---

## Example

Run the example script to see everything in action:

```bash
python example.py
```

This will:

1. Generate attribution maps on dummy data
2. Certify attributions with randomized smoothing
3. Evaluate faithfulness

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
    title={Certified Pixel Attribution for Medical Image Explainability},
    author={Your Name},
    journal={Journal/Conference},
    year={2024}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Contact & Support

For issues or questions:

1. Check the README.md for detailed documentation
2. Review the example.py for usage patterns
3. Check individual module docstrings for API details
