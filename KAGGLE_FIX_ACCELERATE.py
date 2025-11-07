# Emergency fix for accelerate version issue
# Run this cell BEFORE importing peft or starting training

print("🔍 Diagnosing accelerate installation issue...")
print()

# Check what version is currently installed
import subprocess
result = subprocess.run(['pip', 'show', 'accelerate'], capture_output=True, text=True)
print("📦 Current accelerate installation:")
print(result.stdout)
print()

# Force reinstall accelerate 0.30.0 with explicit upgrade
print("🔧 Force reinstalling accelerate 0.30.0...")
print()

# Method 1: Uninstall completely first
subprocess.run(['pip', 'uninstall', '-y', 'accelerate'], check=False)

# Method 2: Install specific version with force-reinstall
subprocess.run([
    'pip', 'install', 
    '--force-reinstall',
    '--no-cache-dir',
    'accelerate==0.30.0'
], check=True)

print()
print("✅ Accelerate reinstalled!")
print()

# Verify installation
result = subprocess.run(['pip', 'show', 'accelerate'], capture_output=True, text=True)
print("📦 New accelerate installation:")
for line in result.stdout.split('\n'):
    if line.startswith('Version:') or line.startswith('Location:'):
        print(f"   {line}")
print()

# Check if clear_device_cache exists in the installed version
print("🔍 Checking if clear_device_cache exists...")
import sys
sys.path.insert(0, '/usr/local/lib/python3.11/dist-packages')

try:
    # Import without loading peft first
    from accelerate.utils.memory import clear_device_cache
    print("✅ clear_device_cache found! You can now import peft.")
except ImportError as e:
    print(f"❌ clear_device_cache still not found: {e}")
    print()
    print("⚠️  This means accelerate 0.30.0 didn't install correctly!")
    print()
    print("🔧 Alternative solution: Use older peft version")
    print("   Run: !pip install peft==0.7.1")
    print("   (But this may have other compatibility issues)")
