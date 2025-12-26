"""
Verification script: check that all modules can be imported and basic operations work.
Run this to verify the project setup is correct.

Usage:
    python verify_setup.py
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def check_imports():
    """Check that all modules can be imported."""
    print("\n" + "="*60)
    print("CHECKING IMPORTS")
    print("="*60)
    
    modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('cv2', 'OpenCV'),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:20} installed")
        except ImportError:
            print(f"✗ {display_name:20} NOT installed")
            all_ok = False
    
    # Check project modules
    print("\nProject modules:")
    project_modules = [
        ('src.models.factory', 'Model factory'),
        ('src.datasets.base', 'Base dataset'),
        ('src.datasets.chexpert', 'CheXpert dataset'),
        ('src.datasets.isic', 'ISIC dataset'),
        ('src.train.train_one', 'Training'),
        ('src.train.metrics', 'Metrics'),
        ('src.xai.attribution', 'Attribution methods'),
        ('src.certify.sparsify', 'Sparsification'),
        ('src.certify.smoothing', 'Randomized smoothing'),
        ('src.certify.evaluate', 'Evaluation'),
        ('src.utils.seed', 'Seed utilities'),
        ('src.utils.io', 'I/O utilities'),
        ('src.utils.viz', 'Visualization'),
    ]
    
    for module_name, display_name in project_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:30} OK")
        except Exception as e:
            print(f"✗ {display_name:30} ERROR: {e}")
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check GPU availability."""
    print("\n" + "="*60)
    print("CHECKING GPU")
    print("="*60)
    
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        return True
    else:
        print("✗ GPU NOT available (CPU only mode)")
        return False


def check_basic_operations():
    """Check that basic operations work."""
    print("\n" + "="*60)
    print("CHECKING BASIC OPERATIONS")
    print("="*60)
    
    import torch
    import numpy as np
    from src.models.factory import get_model
    from src.xai.attribution import IntegratedGradients
    from src.certify.sparsify import sparsify_topk
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Test model loading
        print("Loading model...", end=" ")
        model, config = get_model('resnet18', num_classes=2, device=device)
        model.eval()
        print("✓")
        
        # Test forward pass
        print("Forward pass...", end=" ")
        test_input = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            output = model(test_input)
        assert output.shape == (1, 2)
        print("✓")
        
        # Test attribution
        print("Attribution generation...", end=" ")
        ig = IntegratedGradients(model, device=device)
        with torch.no_grad():
            attr = ig.attribute(test_input, target_class=1, num_steps=10)
        assert attr.shape == (224, 224)
        print("✓")
        
        # Test sparsification
        print("Sparsification...", end=" ")
        sparse = sparsify_topk(attr, k_percent=30)
        assert sparse.shape == attr.shape
        assert np.all((sparse == 0) | (sparse == 1))
        print("✓")
        
        print("\n✓ All basic operations working!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_directory_structure():
    """Check that all required directories exist."""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    required_dirs = [
        'src/datasets',
        'src/models',
        'src/train',
        'src/xai',
        'src/certify',
        'src/experiments',
        'src/utils',
        'configs',
        'data',
        'outputs',
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name:30}")
        else:
            print(f"✗ {dir_name:30} MISSING")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("CERTIFIED PIXEL ATTRIBUTION - SETUP VERIFICATION")
    print("="*60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Python Imports", check_imports),
        ("GPU Availability", check_gpu),
        ("Basic Operations", check_basic_operations),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n✗ {check_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {check_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL CHECKS PASSED - Project is ready to use!")
        print("="*60)
        print("\nNext steps:")
        print("1. Download medical imaging datasets (CheXpert, ISIC, etc.)")
        print("2. Run: python src/experiments/run_train.py --dataset chexpert")
        print("3. Check QUICKSTART.md for detailed workflow")
        print("4. Run: python example.py for a quick demo")
    else:
        print("\n" + "="*60)
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
