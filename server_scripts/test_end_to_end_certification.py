#!/usr/bin/env python
"""Full end-to-end test of grid certification with visualization."""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.grid_dataset import GridDataset
from src.models.grid_multihead import GridMultiHead
from src.xai.attribution_unified import IntegratedGradientsUnified
from src.certify.smoothing import RandomizedSmoothingAttributor

# Load grid
grid_ds = GridDataset(Path("data/raw/grid/isic/val/grid.pt"))
sample = grid_ds[0]
image = sample["image"].unsqueeze(0).cuda()  # [1, 3, 448, 448]
target_class = sample["target_class"] if isinstance(sample["target_class"], int) else int(sample["target_class"].item())
target_head = sample["target_head"] if isinstance(sample["target_head"], int) else int(sample["target_head"].item())

print(f"Image shape: {image.shape}")
print(f"Target class: {target_class}")
print(f"Target head: {target_head}")

# Create model
model = GridMultiHead("resnet18", num_classes=8, num_heads=4, pretrained=False).cuda()
model.eval()

# Create DiFull wrapper
class DiFull_Wrapper(torch.nn.Module):
    def __init__(self, grid_model, head_id, target_cell, scale):
        super().__init__()
        self.grid_model = grid_model
        self.head_id = head_id
        self.target_cell = target_cell
        self.scale = scale
        self.row = target_cell // scale
        self.col = target_cell % scale

    def forward(self, x):
        B, C, full_H, full_W = x.shape
        cell_H = full_H // self.scale
        cell_W = full_W // self.scale
        
        y0, y1 = self.row * cell_H, (self.row + 1) * cell_H
        x0, x1 = self.col * cell_W, (self.col + 1) * cell_W
        
        cell = x[:, :, y0:y1, x0:x1]
        features = self.grid_model.feature_extractor(cell)
        logits = self.grid_model.heads[self.head_id](features)
        
        return logits

wrapper = DiFull_Wrapper(model, target_head, target_head, scale=2)

# Create attribution method
attr_method = IntegratedGradientsUnified(wrapper, device='cuda')

# Test attribution
print("\n" + "="*60)
print("Testing Attribution")
print("="*60)

heat = attr_method.attribute(image, target_class=target_class)
print(f"Heatmap shape: {heat.shape}")
print(f"Heatmap stats:")
print(f"  Cell 0 [0:224, 0:224]: mean={heat[0:224, 0:224].mean():.6f}, max={heat[0:224, 0:224].max():.6f}")
print(f"  Cell 1 [0:224, 224:448]: mean={heat[0:224, 224:448].mean():.6f}, max={heat[0:224, 224:448].max():.6f}")
print(f"  Cell 2 [224:448, 0:224]: mean={heat[224:448, 0:224].mean():.6f}, max={heat[224:448, 0:224].max():.6f}")
print(f"  Cell 3 [224:448, 224:448]: mean={heat[224:448, 224:448].mean():.6f}, max={heat[224:448, 224:448].max():.6f}")

# Test certification
print("\n" + "="*60)
print("Testing Certification (K=50%)")
print("="*60)

smoother = RandomizedSmoothingAttributor(None, None, device='cuda')

def attr_func(img, target_class_override=None):
    tc = target_class if target_class_override is None else int(target_class_override)
    return attr_method.attribute(img, target_class=tc)

smoother.attribution_func = attr_func

results = smoother.certify(
    image,
    k_percent=50,
    target_class=target_class,
    sigma=0.15,
    num_samples=10,  # Small for speed
    tau=0.75,
    batch_size=2,
    alpha=0.001,
)

certified_map = results['certified_map']
print(f"Certified map shape: {certified_map.shape}")
print(f"Certified map stats:")
print(f"  Cell 0 certified pixels: {(certified_map[0:224, 0:224] == 1).sum()}")
print(f"  Cell 1 certified pixels: {(certified_map[0:224, 224:448] == 1).sum()}")
print(f"  Cell 2 certified pixels: {(certified_map[224:448, 0:224] == 1).sum()}")
print(f"  Cell 3 certified pixels: {(certified_map[224:448, 224:448] == 1).sum()}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Input
img_np = image[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
axes[0].imshow(img_np)
axes[0].set_title('Input Grid')
axes[0].axis('off')

# Heatmap
axes[1].imshow(heat, cmap='hot')
axes[1].set_title('Attribution Heatmap')
axes[1].axis('off')

# Certified map
viz_map = np.ones((448, 448, 3))
viz_map[certified_map == 1] = [1.0, 0.65, 0.0]  # orange
viz_map[certified_map == 0] = [1.0, 1.0, 1.0]    # white
viz_map[certified_map == -1] = [0.85, 0.85, 0.85]  # gray
axes[2].imshow(viz_map)
axes[2].set_title('Certified Map')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('test_certification_output.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization to: test_certification_output.png")

if (certified_map[0:224, 0:224] == 1).sum() > 0 and (certified_map[224:448, :] == 1).sum() == 0:
    print("\n✅ SUCCESS: Only Cell 0 is certified!")
else:
    print("\n❌ FAIL: Other cells are also certified!")
