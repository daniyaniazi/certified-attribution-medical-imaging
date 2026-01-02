# Evaluation Module (`src/certify/eval/`)

Clean, reusable evaluation framework for certified attributions. Replicates paper sections 7.2-7.5.

## Architecture

**Inheritance hierarchy:**

```
BaseEvaluator (abstract)
├── RobustnessEvaluator     (7.2: %certified metric)
├── LocalizationEvaluator   (7.3: CertifiedGridPG metric)
└── FaithfulnessEvaluator   (7.5: deletion-based AUC metric)
```

## Module Contents

### `base.py` - BaseEvaluator

Abstract base class providing common infrastructure for all evaluators:

**Constructor:**

```python
BaseEvaluator(dataset_name: str, model_name: str, checkpoint_dir: Path, device='cuda')
```

**Common Methods:**

- `_load_model()` → Loads checkpoint from `checkpoint_dir/[model_name]/{checkpoint_name}.pt`
- `load_cert_results(cert_pkl)` → Deserializes pickle with certification results
- `save_results_json(results, path)` → Saves results as JSON (human-readable)
- `save_results_pkl(results, path)` → Saves results as pickle (reloadable)
- `get_result_paths(output_dir)` → Returns standardized directory structure:
  ```python
  {
      'results_json': output_dir / 'eval_results.json',
      'results_pkl': output_dir / 'eval_results.pkl',
      'figures_dir': output_dir / 'figures',
      'data_dir': output_dir / 'data'
  }
  ```

**Abstract Methods (implement in subclasses):**

- `evaluate_batch(cert_pkl, dataset, output_dir, **kwargs)` → Run evaluation
- `plot_results(results, output_dir)` → Generate visualizations

---

### `robustness.py` - RobustnessEvaluator

**Paper Section:** 7.2 - Robustness Analysis

**Metric:** `%certified` - proportion of images where model's prediction is certifiably robust

```python
evaluator = RobustnessEvaluator('isic', 'resnet18', Path('outputs/checkpoints/isic'))
results = evaluator.evaluate_batch(
    cert_pkl=Path('outputs/bulk_certifcation/isic/resnet18/results_*.pkl'),
    dataset=None,  # Dataset not needed for robustness
    output_dir=Path('outputs/eval/robustness/isic/resnet18')
)
evaluator.save_results_json(results, Path('outputs/eval/.../robustness_results.json'))
evaluator.plot_results(results, Path('outputs/eval/.../figures'))
```

**Output Structure:**

```python
{
    'resnet18': {  # model
        'IntegratedGradients': {  # method
            50: {  # k_percent
                'pct_certified': 0.85,
                'pct_abstained': 0.15,
                'pct_certified_1': 0.42,
                'pct_certified_0': 0.43,
                'certified_radius': 0.025,
                'num_images': 100
            },
            25: {...},
            5: {...}
        },
        'GradCAM': {...},
        ...
    }
}
```

**Figures Generated:**

- `robustness_k50.png` - Bar chart of %certified across methods for K=50
- `robustness_k25.png` - Bar chart of %certified across methods for K=25
- `robustness_k5.png` - Bar chart of %certified across methods for K=5

---

### `localization.py` - LocalizationEvaluator

**Paper Section:** 7.3 - Localization Analysis

**Metric:** `CertifiedGridPG` - certified-region based pointing game score

```python
evaluator = LocalizationEvaluator('isic', 'resnet18', Path('outputs/checkpoints/isic'))

# With ground truth masks
gt_masks = {...}  # {image_idx -> mask}
results = evaluator.evaluate_batch(
    cert_pkl=Path('outputs/bulk_certifcation/isic/resnet18/results_*.pkl'),
    dataset=None,
    output_dir=Path('outputs/eval/localization/isic/resnet18'),
    gt_masks=gt_masks  # Optional: enables GridPG computation
)

# Without GT masks (evaluation skipped gracefully)
results = evaluator.evaluate_batch(
    cert_pkl=...,
    dataset=None,
    output_dir=...,
    gt_masks=None  # Skip localization
)
```

**Output Structure:**

```python
{
    'IntegratedGradients': {  # method
        50: {  # k_percent
            'mean_gridpg': 0.73,
            'std_gridpg': 0.12,
            'num_images': 45,  # only images with GT masks
        },
        25: {...},
        5: {...}
    },
    'GradCAM': {...},
    ...
}
```

**Figures Generated:**

- `localization_k50.png` - Bar chart of GridPG across methods
- `localization_k25.png`
- `localization_k5.png`

---

### `faithfulness.py` - FaithfulnessEvaluator

**Paper Section:** 7.5 - Faithfulness Analysis

**Metric:** `Deletion AUC` - area under curve from iteratively deleting certified pixels

**Key Feature:** Loads certified maps **directly from pickle entries** without requiring `.npy` files.

```python
evaluator = FaithfulnessEvaluator('isic', 'resnet18', Path('outputs/checkpoints/isic'))
results = evaluator.evaluate_batch(
    cert_pkl=Path('outputs/bulk_certifcation/isic/resnet18/results_*.pkl'),
    dataset=None,  # Dataset not needed (uses saved maps)
    output_dir=Path('outputs/eval/faithfulness/isic/resnet18'),
    deletion_steps=5  # Number of deletion iterations
)
evaluator.save_results_json(results, ...)
evaluator.plot_results(results, ...)
```

**Process:**

1. For each image:
   - Extract certified_map from `entry['results']['certified_map']`
   - Identify all "certified-1" pixels (where model prediction remains stable)
   - Iteratively delete certified-1 pixels in groups
   - Measure confidence drop at each deletion step
   - Compute AUC from confidence drop curve

**Output Structure:**

```python
{
    'IntegratedGradients': {  # method
        50: {  # k_percent
            'mean_auc': 0.62,
            'std_auc': 0.15,
            'num_images': 100,
            'mean_baseline_conf': 0.92
        },
        25: {...},
        5: {...}
    },
    'GradCAM': {...},
    ...
}
```

**Figures Generated:**

- `faithfulness_k50.png` - Bar chart of AUC across methods
- `faithfulness_k25.png`
- `faithfulness_k5.png`

---

### `__init__.py` - Module Exports

```python
from src.certify.eval import (
    BaseEvaluator,
    RobustnessEvaluator,
    LocalizationEvaluator,
    FaithfulnessEvaluator,
)
```

---

## Usage Examples

### Single Dataset/Model Evaluation

```python
from pathlib import Path
from src.certify.eval import RobustnessEvaluator, FaithfulnessEvaluator

# Robustness
rob_eval = RobustnessEvaluator('isic', 'resnet18', Path('outputs/checkpoints/isic'))
rob_results = rob_eval.evaluate_batch(
    Path('outputs/bulk_certifcation/isic/resnet18/results_20251230_120000.pkl'),
    None,
    Path('outputs/eval/robustness/isic/resnet18')
)
rob_eval.save_results_json(rob_results, Path('.../robustness_results.json'))
rob_eval.plot_results(rob_results, Path('.../figures'))

# Faithfulness
faith_eval = FaithfulnessEvaluator('isic', 'resnet18', Path('outputs/checkpoints/isic'))
faith_results = faith_eval.evaluate_batch(
    Path('outputs/bulk_certifcation/isic/resnet18/results_20251230_120000.pkl'),
    None,
    Path('outputs/eval/faithfulness/isic/resnet18')
)
faith_eval.save_results_json(faith_results, Path('.../faithfulness_results.json'))
faith_eval.plot_results(faith_results, Path('.../figures'))
```

### Batch Evaluation (All Datasets/Models)

Use the provided runner scripts:

```bash
# Robustness + Faithfulness on all 4 datasets × resnet18
python run_faithfulness_eval.py \
  --cert_base outputs/bulk_certifcation \
  --checkpoint_dir outputs/checkpoints \
  --output_base outputs/eval

# Single dataset/model
python run_evaluation.py \
  --results_pkl outputs/bulk_certifcation/isic/resnet18/results_20251230_120000.pkl \
  --output_dir outputs/eval/isic_resnet18 \
  --dataset isic

# Unified evaluation (all 3 metrics + all datasets)
python run_eval_all.py \
  --evals robustness faithfulness \
  --datasets isic chestxray brain_mri fundus \
  --models resnet18
```

---

## Output Directory Structure

```
outputs/eval/
├── robustness/
│   ├── isic/resnet18/
│   │   ├── robustness_results.json
│   │   ├── robustness_results.pkl
│   │   ├── figures/
│   │   │   ├── robustness_k50.png
│   │   │   ├── robustness_k25.png
│   │   │   └── robustness_k5.png
│   │   └── data/
│   ├── chestxray/resnet18/
│   └── ...
├── localization/
│   └── [dataset]/[model]/...
├── faithfulness/
│   ├── isic/resnet18/
│   │   ├── faithfulness_results.json
│   │   ├── faithfulness_results.pkl
│   │   ├── figures/
│   │   │   ├── faithfulness_k50.png
│   │   │   ├── faithfulness_k25.png
│   │   │   └── faithfulness_k5.png
│   │   └── data/
│   ├── chestxray/resnet18/
│   └── ...
```

---

## Key Design Patterns

### 1. **Template Method Pattern**

`BaseEvaluator` defines the overall structure; subclasses override specific methods:

- `_load_model()` - inherited (same for all)
- `load_cert_results()` - inherited (same for all)
- `evaluate_batch()` - **overridden** per metric
- `plot_results()` - **overridden** per metric

### 2. **Composition over Inheritance**

- Models loaded separately (not inherited)
- Dataset handling optional (passed as parameter)
- Results saved consistently via inherited methods

### 3. **Separation of Concerns**

- **Base class** → infrastructure (loading, I/O, device management)
- **Robustness** → %certified calculation
- **Localization** → CertifiedGridPG with optional GT masks
- **Faithfulness** → deletion curves from saved maps

---

## Future Extensions

To add a new evaluator:

```python
# 1. Create src/certify/eval/[my_metric].py
from src.certify.eval.base import BaseEvaluator

class MyMetricEvaluator(BaseEvaluator):
    def evaluate_batch(self, cert_pkl, dataset, output_dir, **kwargs):
        # Your metric computation
        return results  # dict[method][k_percent][metric]

    def plot_results(self, results, output_dir):
        # Your visualization

# 2. Add to src/certify/eval/__init__.py
from .my_metric import MyMetricEvaluator
__all__ = [..., 'MyMetricEvaluator']

# 3. Use in runner scripts
evaluator = MyMetricEvaluator(dataset_name, model_name, checkpoint_dir)
results = evaluator.evaluate_batch(cert_pkl, dataset, output_dir)
```

---

## Testing

Quick validation:

```python
from pathlib import Path
from src.certify.eval import RobustnessEvaluator

# Test instantiation
eval = RobustnessEvaluator('isic', 'resnet18', Path('outputs/checkpoints/isic'))
print(f"✓ Model loaded: {eval.model}")

# Test loading cert results
cert_pkl = Path('outputs/bulk_certifcation/isic/resnet18/results_*.pkl')
results_dict = eval.load_cert_results(cert_pkl)
print(f"✓ Loaded {len(results_dict)} certification results")
```
