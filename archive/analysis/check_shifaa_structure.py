#!/usr/bin/env python3
"""
Quick diagnostic script to check Shifaa dataset structure
"""

from datasets import load_dataset

print("=" * 80)
print("🔍 CHECKING SHIFAA DATASETS STRUCTURE")
print("=" * 80)
print()

# Check Shifaa Medical
print("1️⃣ Shifaa Medical Consultations:")
print("-" * 80)
try:
    dataset = load_dataset("Ahmed-Selem/Shifaa_Arabic_Medical_Consultations")
    train = dataset['train']
    
    print(f"✅ Loaded: {len(train):,} examples")
    print()
    print("📋 First example:")
    first = train[0]
    print(f"   Keys: {list(first.keys())}")
    print()
    for key in first.keys():
        value = str(first[key])[:100]
        print(f"   {key}: {value}...")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
print()

# Check Shifaa Mental Health
print("2️⃣ Shifaa Mental Health Consultations:")
print("-" * 80)
try:
    dataset = load_dataset("Ahmed-Selem/Shifaa_Arabic_Mental_Health_Consultations")
    train = dataset['train']
    
    print(f"✅ Loaded: {len(train):,} examples")
    print()
    print("📋 First example:")
    first = train[0]
    print(f"   Keys: {list(first.keys())}")
    print()
    for key in first.keys():
        value = str(first[key])[:100]
        print(f"   {key}: {value}...")
    print()
except Exception as e:
    print(f"❌ Error: {e}")
print()

print("=" * 80)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 80)
