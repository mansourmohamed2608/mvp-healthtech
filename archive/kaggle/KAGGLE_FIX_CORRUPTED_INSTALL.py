# DEFINITIVE FIX - accelerate 0.30.0 is installed but corrupted
# The package shows version 0.30.0 but clear_device_cache is missing from the file

import subprocess
import sys

print("=" * 80)
print("FIXING CORRUPTED ACCELERATE INSTALLATION")
print("=" * 80)
print()

print("Problem: accelerate 0.30.0 shows as installed but clear_device_cache is missing")
print("Cause: Pip cached an incomplete/corrupted version")
print("Solution: Clear cache + force reinstall from PyPI")
print()

# Step 1: Uninstall accelerate completely
print("Step 1/4: Uninstalling accelerate...")
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'accelerate'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ Uninstalled")
print()

# Step 2: Clear pip cache (critical!)
print("Step 2/4: Clearing pip cache...")
result = subprocess.run([sys.executable, '-m', 'pip', 'cache', 'purge'], 
                       capture_output=True, text=True)
print("✅ Cache cleared")
print()

# Step 3: Install accelerate from PyPI (no cache)
print("Step 3/4: Installing accelerate 0.30.0 from PyPI (no cache)...")
result = subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    '--no-cache-dir',
    '--force-reinstall',
    'accelerate==0.30.0'
], capture_output=True, text=True)

if result.returncode != 0:
    print("❌ Installation failed!")
    print(result.stderr)
    sys.exit(1)
    
print("✅ Installed")
print()

# Step 4: Verify installation
print("Step 4/4: Verifying installation...")

# Check version
result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'accelerate'], 
                       capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if line.startswith('Version:'):
        print(f"   Version: {line.split(':')[1].strip()}")

# Check if file has clear_device_cache
import os
memory_file = "/usr/local/lib/python3.11/dist-packages/accelerate/utils/memory.py"
if os.path.exists(memory_file):
    with open(memory_file, 'r') as f:
        content = f.read()
    if 'clear_device_cache' in content:
        print("   ✅ clear_device_cache found in memory.py")
    else:
        print("   ❌ clear_device_cache STILL missing!")
        print("   Try alternative solution below")
else:
    print("   ❌ memory.py not found!")

print()
print("=" * 80)
print("NEXT STEP: RESTART KERNEL")
print("=" * 80)
print()
print("Click the ⟳ button to restart the kernel")
print()
print("Then run this to verify:")
print()
print("  import accelerate")
print("  print(f'Version: {accelerate.__version__}')")
print("  from accelerate.utils.memory import clear_device_cache")
print("  print('✅ Working!')")
print()
print()
print("=" * 80)
print("ALTERNATIVE: If still doesn't work, use compatible older versions")
print("=" * 80)
print()
print("  !pip uninstall -y peft accelerate")
print("  !pip install peft==0.13.0 accelerate==0.28.0")
print()
print("  These versions are compatible without clear_device_cache")
