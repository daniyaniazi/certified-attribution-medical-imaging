"""Base dataset class for unified medical imaging interface."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms


class BaseDataset(Dataset, ABC):
    """
    Base class for all medical imaging datasets.
    
    Every dataset should return:
    - image: torch.Tensor [C, H, W]
    - label: int
    - meta: dict with 'id', 'filename', optional 'mask'
    """
    
    def __init__(
        self,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224),
        num_classes: int = 2
    ):
        """
        Initialize dataset.
        
        Args:
            split: 'train', 'val', or 'test'
            transform: torchvision transform
            target_size: (H, W) for resizing
            num_classes: number of output classes
        """
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.num_classes = num_classes
        
        # Subclasses should set these
        self.samples = []  # List of sample metadata
        self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """Load data paths and labels. Must set self.samples."""
        pass
    
    @abstractmethod
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from path as numpy array [H, W] or [H, W, C]."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return single sample."""
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])  # [H, W] or [H, W, 3]
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert grayscale to 3-channel if needed
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        
        # Apply transforms
        if self.transform:
            # Convert back to PIL Image for torchvision transforms
            image = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image)
        else:
            # Default: convert to tensor and move channels first
            image = torch.from_numpy(image).permute(2, 0, 1)
        
        meta = {
            'id': sample.get('id', idx),
            'filename': sample.get('filename', ''),
        }
        if 'mask' in sample and sample['mask'] is not None:
            meta['mask'] = sample['mask']
        
        return {
            'image': image,
            'label': sample['label'],
            'meta': meta
        }
    
    @staticmethod
    def get_default_transform(
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> transforms.Compose:
        """Get default preprocessing pipeline."""
        tf = [
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ]
        
        if normalize:
            tf.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(tf)
