# Check if clear_device_cache actually exists in the installed accelerate
# This will tell us if the package files are corrupted

import os

print("🔍 Checking accelerate installation integrity...")
print()

# Find the accelerate package location
accelerate_path = "/usr/local/lib/python3.11/dist-packages/accelerate"

# Check if the file exists
memory_file = os.path.join(accelerate_path, "utils", "memory.py")
print(f"📂 Checking: {memory_file}")

if os.path.exists(memory_file):
    print("✅ File exists")
    print()
    
    # Read the file and check for clear_device_cache
    with open(memory_file, 'r') as f:
        content = f.read()
    
    if 'clear_device_cache' in content:
        print("✅ clear_device_cache function found in file!")
        print()
        
        # Show where it's defined
        for i, line in enumerate(content.split('\n'), 1):
            if 'def clear_device_cache' in line:
                print(f"   Found at line {i}: {line.strip()}")
        
        print()
        print("⚠️  File has the function but Python can't import it!")
        print("   This suggests a corrupted installation or import cache issue.")
        print()
        print("🔧 SOLUTION: Force reinstall accelerate with cache clear")
        
    else:
        print("❌ clear_device_cache function NOT found in file!")
        print()
        print("   This means accelerate 0.30.0 didn't install correctly.")
        print("   The package might be corrupted or pip cached an old version.")
        print()
        print("🔧 SOLUTION: Force reinstall with --no-cache-dir")
else:
    print("❌ File does not exist!")
    print("   accelerate package is corrupted or incomplete")

print()
print("=" * 80)
print("FIX: Run this in a new cell")
print("=" * 80)
print()
print("!pip uninstall -y accelerate")
print("!pip cache purge")
print("!pip install --no-cache-dir --force-reinstall accelerate==0.30.0")
print()
print("Then restart kernel and test again")
