#!/usr/bin/env python
"""Test if gradient routing through slicing works correctly."""

import torch
import torch.nn as nn

# Test 1: Simple slicing gradient test
print("=" * 60)
print("Test 1: Gradient routing through slicing")
print("=" * 60)

# Create a full grid
full_grid = torch.randn(1, 3, 448, 448, requires_grad=True)

# Extract cell 0 (top-left 224x224)
cell = full_grid[:, :, 0:224, 0:224]

# Simple operation on cell
output = cell.mean()

# Backward
output.backward()

# Check gradients
grad = full_grid.grad
print(f"Full grid shape: {full_grid.shape}")
print(f"Extracted cell shape: {cell.shape}")
print(f"Gradient shape: {grad.shape}")
print(f"\nGradient in Cell 0 region [0:224, 0:224]:")
print(f"  Mean: {grad[0, 0, 0:224, 0:224].mean().item():.6f}")
print(f"  Max: {grad[0, 0, 0:224, 0:224].max().item():.6f}")
print(f"\nGradient in Cell 1 region [0:224, 224:448]:")
print(f"  Mean: {grad[0, 0, 0:224, 224:448].mean().item():.6f}")
print(f"  Max: {grad[0, 0, 0:224, 224:448].max().item():.6f}")
print(f"\nGradient in Cell 2 region [224:448, 0:224]:")
print(f"  Mean: {grad[0, 0, 224:448, 0:224].mean().item():.6f}")
print(f"  Max: {grad[0, 0, 224:448, 0:224].max().item():.6f}")
print(f"\nGradient in Cell 3 region [224:448, 224:448]:")
print(f"  Mean: {grad[0, 0, 224:448, 224:448].mean().item():.6f}")
print(f"  Max: {grad[0, 0, 224:448, 224:448].max().item():.6f}")

if grad[0, 0, 0:224, 0:224].abs().mean() > 0 and grad[0, 0, 224:448, 224:448].abs().mean() == 0:
    print("\n✅ PASS: Gradients correctly localized to Cell 0")
else:
    print("\n❌ FAIL: Gradients leaked to other cells!")

print("\n" + "=" * 60)
print("Test 2: Through a model")
print("=" * 60)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 8)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

model = SimpleModel()

# Full grid
full_grid2 = torch.randn(1, 3, 448, 448, requires_grad=True)

# Extract cell
cell2 = full_grid2[:, :, 0:224, 0:224]

# Forward through model
logits = model(cell2)
target_logit = logits[0, 0]

# Backward
target_logit.backward()

# Check gradients
grad2 = full_grid2.grad
print(f"\nGradient in Cell 0 region: mean={grad2[0, 0, 0:224, 0:224].abs().mean().item():.6f}")
print(f"Gradient in other regions: mean={grad2[0, 0, 224:448, :].abs().mean().item():.6f}")

if grad2[0, 0, 0:224, 0:224].abs().mean() > 0 and grad2[0, 0, 224:448, :].abs().mean() == 0:
    print("\n✅ PASS: Model gradients correctly localized")
else:
    print("\n❌ FAIL: Model gradients leaked!")

print("\n" + "=" * 60)
print("Test 3: DiFull_Wrapper simulation")
print("=" * 60)

class DiFull_Test(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Extract cell 0
        cell = x[:, :, 0:224, 0:224]
        return self.model(cell)

wrapper = DiFull_Test(model)

# Full grid
full_grid3 = torch.randn(1, 3, 448, 448, requires_grad=True)

# Forward through wrapper
logits3 = wrapper(full_grid3)
target_logit3 = logits3[0, 0]

# Backward
target_logit3.backward()

# Check gradients
grad3 = full_grid3.grad
print(f"\nGradient in Cell 0 region: mean={grad3[0, 0, 0:224, 0:224].abs().mean().item():.6f}")
print(f"Gradient in other regions: mean={grad3[0, 0, 224:448, :].abs().mean().item():.6f}")

if grad3[0, 0, 0:224, 0:224].abs().mean() > 0 and grad3[0, 0, 224:448, :].abs().mean() == 0:
    print("\n✅ PASS: DiFull wrapper gradients correctly localized")
else:
    print("\n❌ FAIL: DiFull wrapper gradients leaked!")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("If all tests pass, gradient routing is working correctly.")
print("If tests fail, the issue is with PyTorch's slicing gradient routing.")
