"""I/O utilities for saving/loading checkpoints, configs, results."""
import os
import json
import yaml
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict, output_path: str):
    """Save config as YAML."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(state: Dict, output_path: str):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state, output_path)
    print(f"Checkpoint saved: {output_path}")


def load_checkpoint(checkpoint_path: str, device='cpu'):
    """Load training checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    return state


def save_attribution(attribution: np.ndarray, output_path: str):
    """Save attribution map as .npy."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, attribution.astype(np.float32))


def load_attribution(npy_path: str) -> np.ndarray:
    """Load attribution from .npy."""
    return np.load(npy_path)


def save_json(data: Dict, output_path: str):
    """Save dict as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(json_path: str) -> Dict:
    """Load JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, output_path: str):
    """Save object as pickle."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(pickle_path: str) -> Any:
    """Load pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)
