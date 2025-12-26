"""ISIC skin lesion dataset loader."""
import os
import numpy as np
import json
from typing import Dict, Tuple
from PIL import Image

from src.datasets.base import BaseDataset


class ISICDataset(BaseDataset):
    """
    ISIC (International Skin Imaging Collaboration) dataset.
    8-class classification: AK, BCC, BKL, DF, MEL, NV, SCC, VASC.
    
    Expected directory structure:
    data/processed/isic/
      ├── train/
      │   ├── images/
      │   └── labels.json
      ├── val/
      │   ├── images/
      │   └── labels.json
      └── test/
          ├── images/
          └── labels.json
    """
    
    def __init__(
        self,
        root_dir: str = 'data/raw/isic',
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.labels_path = os.path.join(self.split_dir, 'labels.json')
        
        super().__init__(
            split=split,
            transform=transform,
            target_size=target_size,
            num_classes=8
        )
    
    def _load_data(self):
        """Load ISIC labels from JSON."""
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
                'label': int(label),  # 0 = benign, 1 = melanoma
                'id': filename,
                'filename': filename
            })
    
    def _load_image(self, path: str):
        """Load RGB skin lesion image."""
        img = Image.open(path).convert('RGB')
        return img
