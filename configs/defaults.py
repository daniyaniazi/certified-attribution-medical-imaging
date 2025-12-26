"""
Configuration files for datasets and experiments.
"""

# Default hyperparameters for certification
CERTIFICATION_CONFIG = {
    'sigma': 0.15,          # Gaussian noise std
    'tau': 0.75,            # Certification threshold
    'num_samples': 100,     # Number of smoothing samples
    'batch_size': 16,       # Batch size for smoothing
    'k_percents': [50, 30, 10]  # K values for sparsification
}

# Attribution method configs
ATTRIBUTION_METHODS = {
    'integrated_gradients': {'num_steps': 50},
    'gradcam': {},
    'rise': {'num_samples': 1000, 'mask_size': 7},
    'occlusion': {'patch_size': 16, 'stride': 8}
}

# Model configs
MODEL_CONFIGS = {
    'resnet18': {'pretrained': True, 'target_layer': 'layer4'},
    'densenet121': {'pretrained': True, 'target_layer': 'features'},
    'efficientnet_b0': {'pretrained': True, 'target_layer': 'features'}
}

# Dataset configs
DATASET_CONFIGS = {
    'chexpert': {
        'root_dir': 'data/raw/chexpert',
        'target_size': (224, 224),
        'num_classes': 2,
        'task': 'pneumonia'
    },
    'isic': {
        'root_dir': 'data/raw/isic',
        'target_size': (224, 224),
        'num_classes': 2
    },
    'aptos': {
        'root_dir': 'data/raw/aptos',
        'target_size': (224, 224),
        'num_classes': 5  # Diabetic retinopathy severity
    }
}

# Training configs
TRAINING_CONFIG = {
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'num_workers': 4,
    'metric_to_track': 'val_auc'
}

# Evaluation configs
EVALUATION_CONFIG = {
    'deletion_steps': 50,
    'faithfulness_threshold': 0.5,
    'localization_threshold': 0.5
}
