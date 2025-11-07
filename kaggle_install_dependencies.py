# ============================================================================
# KAGGLE NOTEBOOK - CELL 1: Install Dependencies
# Copy this cell and run FIRST before the main pipeline
# ============================================================================

print("=" * 80)
print("INSTALLING DEPENDENCIES FOR KAGGLE")
print("=" * 80)
print()

# Uninstall problematic packages first
print("🧹 Cleaning up existing packages...")
!pip uninstall -y transformers whisperx -q

print("✅ Cleanup complete")
print()

# Fix numpy/scipy compatibility issue
print("📦 Step 1/4: Fixing numpy/scipy compatibility...")
!pip install -q --upgrade numpy==1.24.3 scipy==1.11.4

print("✅ numpy and scipy upgraded")
print()

# Install PyTorch (if not already installed)
print("📦 Step 2/4: Verifying PyTorch...")
try:
    import torch
    print(f"✅ PyTorch already installed: {torch.__version__}")
except:
    print("Installing PyTorch...")
    !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    print("✅ PyTorch installed")
print()

# Install ML packages with correct versions
print("📦 Step 3/4: Installing Transformers & dependencies...")
!pip install -q transformers==4.44.0 bitsandbytes==0.43.0 accelerate==0.33.0

print("✅ Transformers installed")
print()

# Install WhisperX (after transformers to avoid conflicts)
print("📦 Step 4/4: Installing WhisperX...")
!pip install -q git+https://github.com/m-bain/whisperx.git

print("✅ All packages installed")
print()

# Verify installation
print("=" * 80)
print("VERIFYING INSTALLATION")
print("=" * 80)
print()

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"❌ Transformers: {e}")

try:
    import whisperx
    print(f"✅ WhisperX: Installed")
except Exception as e:
    print(f"❌ WhisperX: {e}")

try:
    import bitsandbytes
    print(f"✅ BitsAndBytes: Installed")
except Exception as e:
    print(f"❌ BitsAndBytes: {e}")

try:
    import numpy as np
    import scipy
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ SciPy: {scipy.__version__}")
except Exception as e:
    print(f"❌ NumPy/SciPy: {e}")

print()
print("=" * 80)
print("INSTALLATION COMPLETE")
print("=" * 80)
print("🚀 Ready to run the main pipeline!")
print("   Copy and paste kaggle_pipeline.py in the next cell")
print("=" * 80)
