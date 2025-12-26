"""
Complete end-to-end example of certified pixel attribution.
This file demonstrates the full workflow from training to certification.

Can be run as a standalone test or imported as a reference.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import get_model
from src.datasets.chexpert import CheXpertDataset
from src.train.train_one import Trainer
from src.xai.attribution import IntegratedGradients
from src.certify.sparsify import sparsify_topk
from src.certify.smoothing import RandomizedSmoothingAttributor
from src.certify.evaluate import CertificationEvaluator
from src.utils.seed import set_seed
from src.utils.viz import save_attribution_heatmap, save_certified_map
import os


def example_training():
    """Example 1: Train a model."""
    print("\n" + "="*60)
    print("EXAMPLE 1: TRAINING")
    print("="*60)
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create dummy dataset (replace with real data)
    print("Loading dataset...")
    try:
        train_dataset = CheXpertDataset(split='train', target_size=(224, 224))
        val_dataset = CheXpertDataset(split='val', target_size=(224, 224))
        
        print(f"Train: {len(train_dataset)} samples")
        print(f"Val: {len(val_dataset)} samples")
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
    except FileNotFoundError:
        print("Note: CheXpert data not found. Skipping training example.")
        print("To run training, download CheXpert dataset from:")
        print("  https://stanfordmlgroup.github.io/competitions/chexpert/")
        return None
    
    # Create model
    print("\nCreating model...")
    model, config = get_model('resnet18', num_classes=2, device=device)
    
    # Train
    print("\nTraining model...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-3,
        task='binary'
    )
    
    # Train for a few epochs (short example)
    history = trainer.fit(
        epochs=2,
        checkpoint_dir='outputs/checkpoints/example',
        metric_to_track='val_auc'
    )
    
    print("\n✓ Training complete!")
    return model


def example_attribution():
    """Example 2: Generate attribution maps."""
    print("\n" + "="*60)
    print("EXAMPLE 2: ATTRIBUTION GENERATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    print("Creating model...")
    model, config = get_model('resnet18', num_classes=2, device=device)
    model.eval()
    
    # Create dummy image
    print("Creating test image...")
    test_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Create attribution method
    print("Generating attributions...")
    ig = IntegratedGradients(model, device=device)
    
    # Compute attribution
    with torch.no_grad():
        attr = ig.attribute(test_image, target_class=1, num_steps=20)
    
    print(f"Attribution shape: {attr.shape}")
    print(f"Attribution range: [{attr.min():.4f}, {attr.max():.4f}]")
    
    # Save visualization
    os.makedirs('outputs/examples', exist_ok=True)
    image_np = test_image[0].permute(1, 2, 0).cpu().numpy()
    save_attribution_heatmap(
        image_np,
        attr,
        'outputs/examples/attribution_example.png',
        title='Example Attribution'
    )
    print("✓ Saved visualization to: outputs/examples/attribution_example.png")
    
    return attr


def example_certification():
    """Example 3: Certify attributions."""
    print("\n" + "="*60)
    print("EXAMPLE 3: CERTIFICATION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    print("Creating model...")
    model, config = get_model('resnet18', num_classes=2, device=device)
    model.eval()
    
    # Create dummy image
    print("Creating test image...")
    test_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Create attribution method
    print("Generating base attribution...")
    ig = IntegratedGradients(model, device=device)
    
    with torch.no_grad():
        attr = ig.attribute(test_image, target_class=1, num_steps=20)
    
    # Sparsify
    print("Sparsifying attribution (K=30%)...")
    sparse_attr = sparsify_topk(attr, k_percent=30)
    print(f"Sparsity: {np.mean(sparse_attr)*100:.1f}%")
    
    # Certify with randomized smoothing
    print("\nCertifying with randomized smoothing...")
    print(f"  sigma=0.15, tau=0.75, num_samples=50")
    
    smoother = RandomizedSmoothingAttributor(model, ig.attribute, device=device)
    
    with torch.no_grad():
        certified, votes, pct_certified = smoother.certify(
            test_image,
            sparse_attr,
            target_class=1,
            sigma=0.15,
            num_samples=50,  # Small number for example
            tau=0.75,
            batch_size=4
        )
    
    print(f"\nResults:")
    print(f"  % Certified: {pct_certified:.1f}%")
    print(f"  Certified to 1: {np.sum(certified==1)}")
    print(f"  Certified to 0: {np.sum(certified==0)}")
    print(f"  Abstained: {np.sum(certified==-1)}")
    
    # Save visualization
    os.makedirs('outputs/examples', exist_ok=True)
    save_certified_map(
        certified,
        'outputs/examples/certified_example.png',
        title='Example Certified Attribution'
    )
    print("\n✓ Saved visualization to: outputs/examples/certified_example.png")
    
    # Evaluate
    print("\nEvaluating...")
    evaluator = CertificationEvaluator()
    metrics = evaluator.evaluate_certified(certified, votes, num_samples=50)
    print(f"  Certified: {metrics['pct_certified']:.1f}%")
    print(f"  Abstained: {metrics['pct_abstained']:.1f}%")
    
    return certified


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CERTIFIED PIXEL ATTRIBUTION - EXAMPLES")
    print("="*60)
    
    # Run examples
    try:
        # Note: Training example requires actual data
        # example_training()
        
        # Attribution generation (works with dummy data)
        attr = example_attribution()
        
        # Certification (works with dummy data)
        certified = example_certification()
        
        print("\n" + "="*60)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
