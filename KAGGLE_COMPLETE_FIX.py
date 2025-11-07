# COMPLETE DIAGNOSTIC AND FIX FOR KAGGLE
# Run this single cell to diagnose and fix the accelerate issue

import subprocess
import sys

print("=" * 80)
print("KAGGLE ACCELERATE FIX - COMPLETE DIAGNOSTIC")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Diagnose current state
# ============================================================================
print("STEP 1: Diagnosing current installation...")
print("-" * 80)

# Check accelerate version on disk
result = subprocess.run(['pip', 'show', 'accelerate'], capture_output=True, text=True)
if result.returncode == 0:
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            accelerate_version = line.split(':')[1].strip()
            print(f"📦 accelerate version on disk: {accelerate_version}")
        if line.startswith('Location:'):
            accelerate_location = line.split(':')[1].strip()
            print(f"📂 accelerate location: {accelerate_location}")
else:
    print("⚠️  accelerate not installed!")
    accelerate_version = None

print()

# Check peft version
result = subprocess.run(['pip', 'show', 'peft'], capture_output=True, text=True)
if result.returncode == 0:
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            peft_version = line.split(':')[1].strip()
            print(f"📦 peft version on disk: {peft_version}")
else:
    print("⚠️  peft not installed!")
    peft_version = None

print()

# Try to import and check what's loaded
print("🔍 Checking what's loaded in Python memory...")
try:
    import accelerate
    print(f"   accelerate in memory: {accelerate.__version__}")
    memory_accelerate = accelerate.__version__
except:
    print("   accelerate not yet imported")
    memory_accelerate = None

print()

# ============================================================================
# STEP 2: Check if clear_device_cache exists
# ============================================================================
print("STEP 2: Checking for clear_device_cache function...")
print("-" * 80)

try:
    from accelerate.utils.memory import clear_device_cache
    print("✅ clear_device_cache found! No fix needed.")
    print()
    print("   Your packages are correctly installed.")
    print("   You can proceed to training.")
    sys.exit(0)
except ImportError as e:
    print(f"❌ clear_device_cache not found!")
    print(f"   Error: {e}")
    print()

# ============================================================================
# STEP 3: Determine the fix needed
# ============================================================================
print("STEP 3: Determining fix strategy...")
print("-" * 80)

if accelerate_version and accelerate_version < "0.30.0":
    print(f"⚠️  Problem: accelerate {accelerate_version} is too old")
    print("   Solution: Upgrade to 0.30.0")
    fix_needed = "upgrade"
elif accelerate_version and accelerate_version >= "0.30.0":
    print(f"⚠️  Problem: accelerate {accelerate_version} installed but not loaded in memory")
    print("   Solution: Restart kernel")
    fix_needed = "restart"
else:
    print("⚠️  Problem: accelerate not installed")
    print("   Solution: Install accelerate 0.30.0")
    fix_needed = "install"

print()

# ============================================================================
# STEP 4: Apply fix
# ============================================================================
print("STEP 4: Applying fix...")
print("-" * 80)

if fix_needed in ["upgrade", "install"]:
    print("🔧 Uninstalling old accelerate...")
    subprocess.run(['pip', 'uninstall', '-y', 'accelerate'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("🔧 Installing accelerate 0.30.0 (force reinstall, no cache)...")
    result = subprocess.run([
        sys.executable, '-m', 'pip', 'install',
        '--force-reinstall',
        '--no-cache-dir',
        '--no-deps',  # Don't install dependencies (they might conflict)
        'accelerate==0.30.0'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Installation failed!")
        print(result.stderr)
    else:
        print("✅ accelerate 0.30.0 installed")
        
        # Now install dependencies separately
        print("🔧 Installing accelerate dependencies...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            'numpy', 'packaging', 'psutil', 'pyyaml', 'torch', 'safetensors', 'huggingface-hub'
        ], check=False)
        
        print("✅ Dependencies installed")
    
    print()
    print("=" * 80)
    print("⚠️  CRITICAL NEXT STEP:")
    print("=" * 80)
    print()
    print("Click the ⟳ RESTART button to reload packages")
    print()
    print("Then run this verification code:")
    print()
    print("    import accelerate")
    print("    print(f'accelerate: {accelerate.__version__}')  # Should be 0.30.0")
    print("    from accelerate.utils.memory import clear_device_cache")
    print("    print('✅ Working!')")
    print()

elif fix_needed == "restart":
    print("=" * 80)
    print("⚠️  SOLUTION:")
    print("=" * 80)
    print()
    print(f"accelerate {accelerate_version} is installed correctly on disk")
    print("But Python has an old version loaded in memory")
    print()
    print("Click the ⟳ RESTART button NOW to reload packages")
    print()
    print("After restart, run:")
    print()
    print("    import accelerate")
    print("    print(f'accelerate: {accelerate.__version__}')  # Should be 0.30.0")
    print("    from accelerate.utils.memory import clear_device_cache")
    print("    print('✅ Working!')")
    print()
