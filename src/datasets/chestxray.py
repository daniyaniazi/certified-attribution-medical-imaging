"""Chest X-ray pneumonia dataset loader."""
import os
import numpy as np
import json
from typing import Dict, Tuple
from PIL import Image

from src.datasets.base import BaseDataset


class ChestXrayDataset(BaseDataset):
    """
    Chest X-ray Pneumonia Dataset (Kaggle).
    2-class classification: NORMAL (0), PNEUMONIA (1).
    
    Expected directory structure:
    data/raw/chestxray/
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
        root_dir: str = 'data/raw/chestxray',
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
            num_classes=2
        )
    
    def _load_data(self):
        """Load ChestXray labels from JSON."""
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
                'label': int(label),  # 0 = NORMAL, 1 = PNEUMONIA
                'id': filename,
                'filename': filename
            })
    
    def _load_image(self, path: str):
        """Load grayscale chest X-ray and convert to RGB."""
        img = Image.open(path).convert('L')  # Load as grayscale
        img_rgb = Image.new('RGB', img.size)
        img_rgb.paste(img)  # Convert to RGB for pretrained backbone
        return img_rgb
