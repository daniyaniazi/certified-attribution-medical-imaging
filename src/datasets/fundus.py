"""Fundus photography (APTOS) diabetic retinopathy dataset loader."""
import os
import json
from typing import Tuple
from PIL import Image

from src.datasets.base import BaseDataset


class FundusDataset(BaseDataset):
    """
    APTOS 2019 Blindness Detection - Fundus Photography Dataset.
    Five-class diabetic retinopathy severity classification.
    
    Classes (0-4):
      0 - No DR (Diabetic Retinopathy)
      1 - Mild DR
      2 - Moderate DR
      3 - Severe DR
      4 - Proliferative DR
    
    Expected directory structure (after preparation):
      data/raw/fundus/
        ├── train/
        │   ├── images/
        │   └── labels.json
        ├── val/
        │   ├── images/
        │   └── labels.json
        └── test/
            ├── images/
            └── labels.json
    
    Each labels.json maps filename -> integer label (0-4).
    """
    
    LABEL_MAP = {
        "no_dr": 0,
        "mild": 1,
        "moderate": 2,
        "severe": 3,
        "proliferative": 4,
    }
    
    CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    def __init__(
        self,
        root_dir: str = 'data/raw/fundus',
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.labels_path = os.path.join(self.split_dir, 'labels.json')
        self.classes = self.CLASS_NAMES
        
        super().__init__(
            split=split,
            transform=transform,
            target_size=target_size,
            num_classes=len(self.LABEL_MAP)
        )
    
    def _load_data(self):
        """Load labels and image paths for the selected split."""
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels not found: {self.labels_path}")
        
        with open(self.labels_path, 'r') as f:
            labels_dict = json.load(f)
        
        images_dir = os.path.join(self.split_dir, 'images')
        
        for filename, label in labels_dict.items():
            image_path = os.path.join(images_dir, filename)
            if not os.path.exists(image_path):
                continue
            self.samples.append({
                'image_path': image_path,
                'label': int(label),
                'id': filename,
                'filename': filename
            })
    
    def _load_image(self, path: str):
        """Load fundus photograph and ensure RGB format."""
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
