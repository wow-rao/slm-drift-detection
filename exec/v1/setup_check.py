#!/usr/bin/env python3
import sys
import os

def check_dependencies():
    print("Checking dependencies...")
    
    deps = {
        'torch': 'PyTorch',
        'transformers': 'Transformers (Hugging Face)',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name} - NOT FOUND")
            missing.append(module)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n[OK] All dependencies installed!")
    return True

def check_cuda():
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available (Device: {torch.cuda.get_device_name(0)})")
        else:
            print("  [WARNING] CUDA not available - will use CPU")
    except Exception as e:
        print(f"  [WARNING] Could not check CUDA: {e}")

def create_directories():
    print("\nCreating required directories...")
    
    dirs = ['data/train', 'data/test', 'logs']
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  [OK] {d}")
    
    print("\n[OK] Directories created!")

def main():
    print("="*60)
    print("Document Anomaly Detection - Setup Verification")
    print("="*60)
    print()
    
    deps_ok = check_dependencies()
    check_cuda()
    create_directories()
    
    print("\n" + "="*60)
    if deps_ok:
        print("[OK] Setup complete! You're ready to go.")
        print("\nNext steps:")
        print("  1. Place your training documents in data/train/")
        print("  2. Place your test documents in data/test/")
        print("  3. Run: python src/exec/v1/train.py --epochs 50 --train_dir data/train")
        print("  4. Run: python src/exec/v1/play.py --test_dir data/test")
        print("\nFor detailed instructions, see: src/exec/v1/README.md")
    else:
        print("[ERROR] Setup incomplete. Please install missing dependencies.")
        sys.exit(1)
    print("="*60)

if __name__ == "__main__":
    main()
