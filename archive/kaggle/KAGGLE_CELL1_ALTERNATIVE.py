# CELL 1 - ALTERNATIVE VERSION (More Aggressive Installation)
# Use this if the regular Cell 1 doesn't work

print("🔧 Installing training dependencies (AGGRESSIVE METHOD)...")
print("⚠️  You'll see dependency warnings - IGNORE THEM")
print()

# Step 1: Uninstall ALL conflicting packages completely
print("Step 1/3: Uninstalling old versions...")
import subprocess
subprocess.run(['pip', 'uninstall', '-y', 'transformers', 'tokenizers', 'accelerate', 'peft', 'bitsandbytes', 'trl'], 
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ Old packages removed")
print()

# Step 2: Install accelerate FIRST with force-reinstall
print("Step 2/3: Installing accelerate 0.30.0 (with force)...")
subprocess.run([
    'pip', 'install',
    '--force-reinstall',
    '--no-cache-dir',
    'accelerate==0.30.0'
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("✅ accelerate 0.30.0 installed")
print()

# Step 3: Install everything else
print("Step 3/3: Installing other packages...")
packages = [
    'transformers==4.36.2',
    'peft==0.15.0',
    'bitsandbytes==0.42.0',
    'trl==0.7.10',
    'datasets==2.16.1',
    'scipy==1.11.4',
    'sentencepiece==0.1.99',
    'protobuf==4.25.1',
    'openpyxl==3.1.2'
]

for pkg in packages:
    subprocess.run(['pip', 'install', '-q', pkg], check=False)
    
print("✅ All packages installed")
print()

# Verify installation
print("🔍 Verifying installation...")
result = subprocess.run(['pip', 'show', 'accelerate'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if line.startswith('Version:'):
        print(f"   accelerate: {line.split(':')[1].strip()}")

result = subprocess.run(['pip', 'show', 'peft'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if line.startswith('Version:'):
        print(f"   peft: {line.split(':')[1].strip()}")

print()
print("✅ Installation complete!")
print()
print("⚠️  CRITICAL: Now click 'Restart Kernel' (⟳ button) to load new versions")
print("   Then run the verification cell below:")
print()
print("   import peft, accelerate")
print("   print(f'peft: {peft.__version__}')")
print("   print(f'accelerate: {accelerate.__version__}')")
print("   from accelerate.utils.memory import clear_device_cache")
print("   print('✅ Working!')")
print()
