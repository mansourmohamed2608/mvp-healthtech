import json

# Load results
with open('wer_comparison_results.json', encoding='utf-8') as f:
    data = json.load(f)

# Get word counts
no_lora_text = data['no_lora']['transcription']['text']
lora_text = data['with_lora']['transcription']['text']

no_lora_words = len(no_lora_text.split())
lora_words = len(lora_text.split())

no_lora_chars = len(no_lora_text)
lora_chars = len(lora_text)

# Get timing info
no_lora_time = data['no_lora']['transcription']['processing_time']
lora_time = data['with_lora']['transcription']['processing_time']

print("=" * 60)
print("📊 LORA vs NO-LORA COMPARISON")
print("=" * 60)
print(f"\n🔤 WORD COUNT:")
print(f"  Without LoRA: {no_lora_words} words")
print(f"  With LoRA:    {lora_words} words")
print(f"  Difference:   {lora_words - no_lora_words} words")

print(f"\n📝 CHARACTER COUNT:")
print(f"  Without LoRA: {no_lora_chars} chars")
print(f"  With LoRA:    {lora_chars} chars")

print(f"\n⏱️  PROCESSING TIME:")
print(f"  Without LoRA: {no_lora_time:.1f}s")
print(f"  With LoRA:    {lora_time:.1f}s")
print(f"  Difference:   {lora_time - no_lora_time:.1f}s slower")
print(f"  Percentage:   {((lora_time / no_lora_time) - 1) * 100:.1f}% slower")

print(f"\n🎯 LORA TEXT PREVIEW (first 300 chars):")
print(f"  {lora_text[:300]}...")

print(f"\n" + "=" * 60)
