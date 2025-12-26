"""CheXpert chest X-ray dataset loader."""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from PIL import Image

from src.datasets.base import BaseDataset


class CheXpertDataset(BaseDataset):
    """
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels.
    
    Expected directory structure:
    data/raw/chexpert/
      ├── train.csv
      ├── valid.csv
      └── [images]/
    """
    
    def __init__(
        self,
        root_dir: str = 'data/raw/chexpert',
        split: str = 'train',
        task: str = 'pneumonia',  # or 'pleural_effusion', 'cardiomegaly', etc.
        transform=None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = root_dir
        self.task = task
        self.csv_path = os.path.join(root_dir, f'{split}.csv')
        
        super().__init__(
            split=split,
            transform=transform,
            target_size=target_size,
            num_classes=2
        )
    
    def _load_data(self):
        """Load CheXpert CSV and prepare samples."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Map task to column
        task_col_map = {
            'pneumonia': 'Pneumonia',
            'pleural_effusion': 'Pleural Effusion',
            'cardiomegaly': 'Cardiomegaly',
            'consolidation': 'Consolidation',
            'edema': 'Edema'
        }
        
        label_col = task_col_map.get(self.task, 'Pneumonia')
        
        for idx, row in df.iterrows():
            path = row['Path']
            full_path = os.path.join(self.root_dir, path)
            
            # Skip if doesn't exist
            if not os.path.exists(full_path):
                continue
            
            # Get label (uncertain labels -> drop)
            label = row[label_col]
            if pd.isna(label) or label == -1:  # -1 = uncertain
                continue
            
            label = int(label)
            
            self.samples.append({
                'image_path': full_path,
                'label': label,
                'id': idx,
                'filename': os.path.basename(path)
            })
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load grayscale chest X-ray."""
        img = Image.open(path).convert('L')
        return np.array(img)
