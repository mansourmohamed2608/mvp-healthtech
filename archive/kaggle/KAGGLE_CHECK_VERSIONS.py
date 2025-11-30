# Quick version check script for Kaggle
# Run this AFTER Cell 1 + Restart to verify packages loaded correctly

print("🔍 Checking installed package versions...")
print()

# Check what's installed on disk
import subprocess
result = subprocess.run(['pip', 'show', 'accelerate'], capture_output=True, text=True)
print("📦 accelerate (on disk):")
for line in result.stdout.split('\n'):
    if line.startswith('Version:'):
        print(f"   {line}")
print()

result = subprocess.run(['pip', 'show', 'peft'], capture_output=True, text=True)
print("📦 peft (on disk):")
for line in result.stdout.split('\n'):
    if line.startswith('Version:'):
        print(f"   {line}")
print()

# Try to import and check loaded versions
print("🔄 Trying to import packages...")
try:
    import accelerate
    print(f"✅ accelerate loaded: {accelerate.__version__}")
except Exception as e:
    print(f"❌ accelerate import failed: {e}")

try:
    import peft
    print(f"✅ peft loaded: {peft.__version__}")
except Exception as e:
    print(f"❌ peft import failed: {e}")

print()
print("🔍 Checking if clear_device_cache exists...")
try:
    from accelerate.utils.memory import clear_device_cache
    print("✅ clear_device_cache imported successfully!")
except ImportError as e:
    print(f"❌ clear_device_cache not found: {e}")
    print()
    print("⚠️  This means accelerate 0.30.0 is NOT loaded yet!")
    print("   Solution: Click 'Restart & Run All' in Kaggle to reload packages")
