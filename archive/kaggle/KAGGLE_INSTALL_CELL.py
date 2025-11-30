# ===================================================================
# KAGGLE TRAINING - WORKING CONFIGURATION (UPDATED - FIX 3)
# ===================================================================
# Copy-paste this entire cell into Kaggle as Cell 1
# Then restart kernel and run your main script

print("Installing dependencies...")
print("=" * 80)

# STEP 1: Uninstall conflicting packages
print("\n1. Cleaning up conflicting packages...")
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", 
                      "transformers", "tokenizers", "huggingface-hub", "safetensors"])

# STEP 2: Install transformers 4.44.0 with dependencies
print("\n2. Installing transformers 4.44.0 with dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.44.0"])

# STEP 3: Install accelerate
print("\n3. Installing accelerate...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.27.0"])

# STEP 4: Install bitsandbytes (CRITICAL - no quiet mode to see errors!)
print("\n4. Installing bitsandbytes...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes>=0.43.0"])

print("\n" + "=" * 80)
print("✅ INSTALLATION COMPLETE!")
print("=" * 80)

# Verification
print("\nVerifying installations...")
try:
    import transformers
    import tokenizers
    import accelerate
    import bitsandbytes
    
    print(f"\n✓ transformers: {transformers.__version__}")
    print(f"✓ tokenizers: {tokenizers.__version__}")
    print(f"✓ accelerate: {accelerate.__version__}")
    print(f"✓ bitsandbytes: {bitsandbytes.__version__}")
    
    # Validate versions
    errors = []
    if transformers.__version__ != "4.44.0":
        errors.append(f"transformers is {transformers.__version__}, expected 4.44.0")
    
    if not tokenizers.__version__.startswith("0.19"):
        errors.append(f"tokenizers is {tokenizers.__version__}, expected 0.19.x")
    
    if errors:
        print("\n⚠️  WARNINGS:")
        for err in errors:
            print(f"   - {err}")
    else:
        print("\n✅ ALL VERSIONS CORRECT!")
    
except ImportError as e:
    print(f"\n❌ ERROR: Failed to import {e.name}")
    print("   Installation may have failed. Check output above for errors.")
    raise

print("\n" + "=" * 80)
print("⚠️  MANDATORY NEXT STEP:")
print("   1. Click 'Kernel' → 'Restart kernel'")
print("   2. Wait for 'Ready' indicator (bottom left)")
print("   3. Then run Cell 2 (main script)")
print("=" * 80)
