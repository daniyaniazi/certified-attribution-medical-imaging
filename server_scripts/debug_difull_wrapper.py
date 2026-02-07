#!/usr/bin/env python
"""Debug: Test DiFull_Wrapper with actual grid and check attribution."""

import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.grid_dataset import GridDataset
from src.models.grid_multihead import GridMultiHead

# Load grid dataset
grid_ds = GridDataset(Path("data/raw/grid/isic/val/grid.pt"))
sample = grid_ds[0]
image = sample["image"].unsqueeze(0)  # [1, 3, 448, 448]
target_class = sample["target_class"] if isinstance(sample["target_class"], int) else int(sample["target_class"].item())
target_head = sample["target_head"] if isinstance(sample["target_head"], int) else int(sample["target_head"].item())

print(f"Test image shape: {image.shape}")
print(f"Target class: {target_class}")
print(f"Target head (cell): {target_head}")

# Create model
model = GridMultiHead("resnet18", num_classes=8, num_heads=4, pretrained=False)
model.eval()

print(f"\n" + "="*60)
print("Test 1: Direct model call (no wrapper)")
print("="*60)

# Test direct model call with full grid
image.requires_grad_(True)
logits_full = model(image, head_id=target_head)
print(f"Logits from full grid: {logits_full[0]}")

target_logit = logits_full[0, target_class]
target_logit.backward()

grad_full = image.grad.clone()
print(f"\nGradient stats:")
print(f"  Cell 0 [0:224, 0:224]: mean={grad_full[0, 0, 0:224, 0:224].abs().mean().item():.6f}")
print(f"  Cell 1 [0:224, 224:448]: mean={grad_full[0, 0, 0:224, 224:448].abs().mean().item():.6f}")
print(f"  Cell 2 [224:448, 0:224]: mean={grad_full[0, 0, 224:448, 0:224].abs().mean().item():.6f}")
print(f"  Cell 3 [224:448, 224:448]: mean={grad_full[0, 0, 224:448, 224:448].abs().mean().item():.6f}")

print(f"\n" + "="*60)
print("Test 2: DiFull_Wrapper")
print("="*60)

# Create DiFull_Wrapper
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
        
        # Extract target cell
        cell = x[:, :, y0:y1, x0:x1]
        
        # Process ONLY the target cell
        features = self.grid_model.feature_extractor(cell)
        logits = self.grid_model.heads[self.head_id](features)
        
        return logits

wrapper = DiFull_Wrapper(model, target_head, target_head, scale=2)

# Test wrapper with full grid
image2 = image.clone().detach()
image2.requires_grad_(True)

logits_wrapper = wrapper(image2)
print(f"Logits from wrapper: {logits_wrapper[0]}")

target_logit2 = logits_wrapper[0, target_class]
target_logit2.backward()

grad_wrapper = image2.grad
print(f"\nGradient stats:")
print(f"  Cell 0 [0:224, 0:224]: mean={grad_wrapper[0, 0, 0:224, 0:224].abs().mean().item():.6f}")
print(f"  Cell 1 [0:224, 224:448]: mean={grad_wrapper[0, 0, 0:224, 224:448].abs().mean().item():.6f}")
print(f"  Cell 2 [224:448, 0:224]: mean={grad_wrapper[0, 0, 224:448, 0:224].abs().mean().item():.6f}")
print(f"  Cell 3 [224:448, 224:448]: mean={grad_wrapper[0, 0, 224:448, 224:448].abs().mean().item():.6f}")

# Check if gradients are localized
cell_0_grad = grad_wrapper[0, :, 0:224, 0:224].abs().mean().item()
other_cells_grad = grad_wrapper[0, :, 224:448, :].abs().mean().item()

if cell_0_grad > 0 and other_cells_grad == 0:
    print("\n✅ PASS: Gradients correctly localized to target cell")
else:
    print(f"\n❌ FAIL: Gradients leaked to other cells!")
    print(f"   Target cell grad: {cell_0_grad:.6f}")
    print(f"   Other cells grad: {other_cells_grad:.6f}")

print(f"\n" + "="*60)
print("Test 3: IntegratedGradients simulation")
print("="*60)

# Simulate IG
baseline = torch.zeros_like(image)
alphas = torch.linspace(0, 1, 11)  # Simplified: only 11 steps
accumulated_grads = None

for alpha in alphas:
    interpolated = (baseline + alpha * (image - baseline)).clone().detach()
    interpolated.requires_grad_(True)
    
    output = wrapper(interpolated)
    logit = output[0, target_class]
    
    grads = torch.autograd.grad(logit, interpolated, create_graph=False)[0]
    
    if accumulated_grads is None:
        accumulated_grads = grads.clone()
    else:
        accumulated_grads += grads

avg_grads = accumulated_grads / len(alphas)
integrated_grads = (image - baseline) * avg_grads.detach()
attribution = integrated_grads.sum(dim=1)[0].abs()

print(f"Attribution shape: {attribution.shape}")
print(f"Attribution stats:")
print(f"  Cell 0 [0:224, 0:224]: mean={attribution[0:224, 0:224].mean().item():.6f}, max={attribution[0:224, 0:224].max().item():.6f}")
print(f"  Cell 1 [0:224, 224:448]: mean={attribution[0:224, 224:448].mean().item():.6f}, max={attribution[0:224, 224:448].max().item():.6f}")
print(f"  Cell 2 [224:448, 0:224]: mean={attribution[224:448, 0:224].mean().item():.6f}, max={attribution[224:448, 0:224].max().item():.6f}")
print(f"  Cell 3 [224:448, 224:448]: mean={attribution[224:448, 224:448].mean().item():.6f}, max={attribution[224:448, 224:448].max().item():.6f}")

if attribution[0:224, 0:224].mean() > 0 and attribution[224:448, :].max() < 1e-6:
    print("\n✅ PASS: Attribution correctly localized to target cell")
else:
    print("\n❌ FAIL: Attribution spread across multiple cells!")
