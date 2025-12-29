"""Brain MRI tumor dataset loader."""
import os
import json
from typing import Tuple
from PIL import Image

from src.datasets.base import BaseDataset


class BrainMRIDataset(BaseDataset):
    """
    Brain MRI tumor classification dataset.
    Four classes: normal, glioma_tumor, meningioma_tumor, pituitary_tumor.

    Expected directory structure (after preparation):
      data/raw/brain_mri/
        ├── train/
        │   ├── images/
        │   └── labels.json
        ├── val/
        │   ├── images/
        │   └── labels.json
        └── test/
            ├── images/
            └── labels.json

    Each labels.json maps filename -> integer label using LABEL_MAP order.
    """

    LABEL_MAP = {
        "normal": 0,
        "glioma_tumor": 1,
        "meningioma_tumor": 2,
        "pituitary_tumor": 3,
    }

    def __init__(
        self,
        root_dir: str = 'data/raw/brain_mri',
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.labels_path = os.path.join(self.split_dir, 'labels.json')
        self.classes = sorted(self.LABEL_MAP, key=self.LABEL_MAP.get)

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
        """Load MRI slice and ensure RGB format for pretrained backbones."""
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
