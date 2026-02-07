"""
Test whether DiFull gradients flow to the FULL image or just the crop.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys
import numpy as np

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

from src.models.grid_multihead import GridMultiHead
import torch.nn as nn


class DiFull_Wrapper_Old(nn.Module):
    """OLD: Crops before backbone - gradients only to target cell."""
    def __init__(self, grid_model, head_id):
        super().__init__()
        self.grid_model = grid_model
        self.head_id = head_id
    
    def forward(self, x):
        return self.grid_model(x, head_id=self.head_id)


class DiFull_Wrapper_New(nn.Module):
    """NEW: Full image through backbone - gradients to all cells."""
    def __init__(self, grid_model, head_id):
        super().__init__()
        self.grid_model = grid_model
        self.head_id = head_id
    
    def forward(self, x):
        # Process FULL image through backbone
        full_features = self.grid_model.feature_extractor(x)
        # Apply target head
        logits = self.grid_model.heads[self.head_id](full_features)
        return logits


def test_gradient_flow():
    """Check if gradients flow to full image or just crop."""
    device = 'cpu'
    
    # Load model
    model = GridMultiHead(num_heads=4, num_classes=2, scale=2).to(device)
    model.eval()
    
    # Create synthetic grid image: 384x384 (2x2 grid of 192x192 cells)
    # Cell 0 (top-left): all red
    # Cell 1 (top-right): all green
    # Cell 2 (bottom-left): all blue
    # Cell 3 (bottom-right): all white
    img_np = np.zeros((3, 384, 384), dtype=np.float32)
    
    # Cell 0 (rows 0-191, cols 0-191): red
    img_np[0, 0:192, 0:192] = 1.0
    
    # Cell 1 (rows 0-191, cols 192-384): green
    img_np[1, 0:192, 192:384] = 1.0
    
    # Cell 2 (rows 192-384, cols 0-191): blue
    img_np[2, 192:384, 0:192] = 1.0
    
    # Cell 3 (rows 192-384, cols 192-384): white
    img_np[:, 192:384, 192:384] = 1.0
    
    img_tensor = torch.from_numpy(img_np)
    
    print(f"Full image shape: {img_tensor.shape}")
    
    # Test for head_id=0 (bottom-left cell in 2x2 grid)
    head_id = 0
    
    # Create input that requires grad
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, 384, 384]
    
    print("\n" + "="*60)
    print("TEST 1: OLD WRAPPER (crop before backbone)")
    print("="*60)
    
    img_tensor_old = img_tensor.clone()
    img_tensor_old.requires_grad_(True)
    
    wrapper_old = DiFull_Wrapper_Old(model, head_id)
    logits_old = wrapper_old(img_tensor_old)
    target_logit_old = logits_old[0, 1]
    target_logit_old.backward()
    
    grad_old = img_tensor_old.grad[0].abs().sum(dim=0)
    
    print(f"Gradient sum (Cell 0): {grad_old[0:192, 0:192].sum().item():.6f}")
    print(f"Gradient sum (Cell 1): {grad_old[0:192, 192:384].sum().item():.6f}")
    print(f"Gradient sum (Cell 2): {grad_old[192:384, 0:192].sum().item():.6f}")
    print(f"Gradient sum (Cell 3): {grad_old[192:384, 192:384].sum().item():.6f}")
    
    print("\n" + "="*60)
    print("TEST 2: NEW WRAPPER (full image through backbone)")
    print("="*60)
    
    img_tensor_new = img_tensor.clone()
    img_tensor_new.requires_grad_(True)
    
    wrapper_new = DiFull_Wrapper_New(model, head_id)
    logits_new = wrapper_new(img_tensor_new)
    target_logit_new = logits_new[0, 1]
    target_logit_new.backward()
    
    grad_new = img_tensor_new.grad[0].abs().sum(dim=0)
    
    print(f"Gradient sum (Cell 0): {grad_new[0:192, 0:192].sum().item():.6f}")
    print(f"Gradient sum (Cell 1): {grad_new[0:192, 192:384].sum().item():.6f}")
    print(f"Gradient sum (Cell 2): {grad_new[192:384, 0:192].sum().item():.6f}")
    print(f"Gradient sum (Cell 3): {grad_new[192:384, 192:384].sum().item():.6f}")
    
    # Visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_tensor[0].detach().permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Full Input Image (4 cells)')
    axes[0].axis('off')
    
    axes[1].imshow(grad_old.cpu().numpy(), cmap='hot')
    axes[1].set_title(f'OLD: Gradients (crop before backbone)')
    axes[1].axis('off')
    
    axes[2].imshow(grad_new.cpu().numpy(), cmap='hot')
    axes[2].set_title(f'NEW: Gradients (full image through backbone)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradient_flow_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison to gradient_flow_comparison.png")


if __name__ == '__main__':
    test_gradient_flow()
