"""
Generate Egyptian Arabic Medical Training Data for Fine-tuning
===============================================================

This script uses GPT-4 to generate high-quality Egyptian dialect medical
conversations and SOAP notes for fine-tuning MMed-Llama-3-8B.

Cost: ~$20-30 for 1000 examples (one-time)
Time: 2-3 hours

Requirements:
    pip install openai datasets

Usage:
    1. Set OPENAI_API_KEY environment variable
    2. Run: python generate_training_data.py
    3. Output: training_data.json (1000 examples)
"""

import os
import json
import time
from openai import OpenAI
from typing import List, Dict
import random

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Medical scenarios to generate
SCENARIOS = [
    # Dental
    "مريض يشتكي من التهاب اللثة ونزيف عند تنظيف الأسنان",
    "مريضة تعاني من ألم في ضرس العقل",
    "طفل عنده تسوس في الأسنان الأمامية",
    
    # General Medicine
    "مريض يشتكي من صداع مستمر منذ أسبوع",
    "مريضة عندها ارتفاع في ضغط الدم",
    "مريض يعاني من آلام في المعدة والإسهال",
    "مريضة تشتكي من ألم في الظهر",
    
    # Respiratory
    "مريض عنده كحة وبلغم منذ أسبوعين",
    "طفلة تعاني من ضيق في التنفس وصفير",
    "مريض يشتكي من التهاب في الحلق وحمى",
    
    # Chronic Conditions
    "مريض سكري يراجع لمتابعة السكر التراكمي",
    "مريضة تعاني من آلام المفاصل والروماتيزم",
    "مريض يشتكي من أرق وصعوبة في النوم",
    
    # Pediatrics
    "طفل رضيع عنده إسهال وقيء",
    "طفلة عندها طفح جلدي وحكة",
    "أم تسأل عن تطعيمات الطفل",
    
    # Women's Health
    "مريضة حامل في الشهر السادس تسأل عن التغذية",
    "مريضة تشتكي من آلام الدورة الشهرية",
    
    # ENT
    "مريض يشتكي من طنين في الأذن",
    "مريضة عندها التهاب في الجيوب الأنفية",
    
    # Dermatology
    "مريض يشتكي من حب الشباب",
    "مريضة تعاني من تساقط الشعر",
]

def generate_conversation(scenario: str) -> Dict:
    """Generate Egyptian dialect medical conversation + SOAP note"""
    
    prompt = f"""You are a medical conversation generator for Egyptian Arabic dialect.

Generate a realistic doctor-patient conversation in Egyptian Arabic for this scenario:
"{scenario}"

Requirements:
1. Use natural Egyptian dialect (ازيك، حاسس ايه، الدكتور، عيان، etc.)
2. Conversation should be 8-12 exchanges (doctor asks, patient responds)
3. Doctor should:
   - Ask about symptoms and history
   - Explain diagnosis clearly in Egyptian dialect
   - Provide treatment plan
   - Give follow-up instructions
4. Patient should use colloquial Egyptian (مش، عندي، بحس، وحشة، etc.)
5. Make it sound natural, not formal

After the conversation, generate a SOAP note in Modern Standard Arabic (MSA).

Format your response EXACTLY like this:

=== CONVERSATION ===
دكتور: ازيك يا فندم؟
مريض: والله مش كويس يا دكتور...
[... rest of conversation ...]

=== SOAP NOTE ===
S (Subjective): ...
O (Objective): ...
A (Assessment): ...
P (Plan): ...

Generate now:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper than gpt-4, still good quality
            messages=[
                {"role": "system", "content": "You are an expert in Egyptian Arabic medical conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,  # More creative for varied conversations
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        
        # Parse the response
        if "=== CONVERSATION ===" in content and "=== SOAP NOTE ===" in content:
            parts = content.split("=== SOAP NOTE ===")
            conversation = parts[0].replace("=== CONVERSATION ===", "").strip()
            soap_note = parts[1].strip()
            
            return {
                "scenario": scenario,
                "conversation": conversation,
                "soap_note": soap_note,
                "model": "gpt-4o-mini"
            }
        else:
            print(f"⚠️  Failed to parse response for scenario: {scenario}")
            return None
            
    except Exception as e:
        print(f"❌ Error generating conversation: {e}")
        return None

def generate_correction_examples() -> List[Dict]:
    """Generate text correction examples"""
    
    # Common ASR errors in Egyptian Arabic
    corrections = [
        {
            "input": "المريض يشتكي من الم في الراس منذ تلاتة ايام",
            "output": "المريض يشتكي من ألم في الرأس منذ ثلاثة أيام"
        },
        {
            "input": "عندي وجع في اللتة وبتنزف لما بغسل سناني",
            "output": "عندي وجع في اللثة وبتنزف لما بغسل سناني"
        },
        {
            "input": "الدكتور قالي لازم اخد المضاد الحيوي تلات مرات في اليوم",
            "output": "الدكتور قالي لازم آخد المضاد الحيوي ثلاث مرات في اليوم"
        },
        # Add 50 more correction examples
    ]
    
    return corrections

def format_for_training(examples: List[Dict]) -> List[Dict]:
    """Format examples for instruction fine-tuning (Alpaca format)"""
    
    formatted = []
    
    for ex in examples:
        if ex is None:
            continue
            
        # Format for instruction tuning
        formatted.append({
            "instruction": "أنت طبيب مساعد. اكتب تقرير SOAP للمحادثة الطبية التالية:",
            "input": ex["conversation"],
            "output": ex["soap_note"],
            "metadata": {
                "scenario": ex["scenario"],
                "task": "soap_generation",
                "dialect": "egyptian"
            }
        })
    
    return formatted

def main():
    print("=" * 80)
    print("GENERATING TRAINING DATA")
    print("=" * 80)
    print()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not set!")
        print("Set it with: $env:OPENAI_API_KEY='your-key-here'")
        return
    
    print(f"Target: 1000 examples")
    print(f"Cost estimate: $20-30")
    print(f"Time estimate: 2-3 hours")
    print()
    
    # Generate conversations for each scenario
    all_examples = []
    
    # Generate multiple variations per scenario
    variations_per_scenario = 50  # 20 scenarios × 50 = 1000 examples
    
    for i, scenario in enumerate(SCENARIOS):
        print(f"Scenario {i+1}/{len(SCENARIOS)}: {scenario}")
        
        for j in range(variations_per_scenario):
            print(f"  Generating variation {j+1}/{variations_per_scenario}...", end="\r")
            
            example = generate_conversation(scenario)
            if example:
                all_examples.append(example)
            
            # Rate limiting (OpenAI: 3 requests/min for free tier)
            time.sleep(0.5)  # Safe rate: 2 req/sec = 120 req/min
        
        print(f"  ✅ Generated {len(all_examples)} examples so far")
    
    print()
    print(f"✅ Generated {len(all_examples)} total examples")
    
    # Format for training
    print("Formatting for instruction tuning...")
    training_data = format_for_training(all_examples)
    
    # Save to file
    output_file = "training_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved to {output_file}")
    print()
    
    # Stats
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total examples: {len(training_data)}")
    print(f"Average conversation length: {sum(len(ex['input']) for ex in training_data) / len(training_data):.0f} chars")
    print(f"Average SOAP note length: {sum(len(ex['output']) for ex in training_data) / len(training_data):.0f} chars")
    print()
    
    # Show sample
    print("Sample example:")
    print("-" * 80)
    print(f"Instruction: {training_data[0]['instruction']}")
    print(f"Input (first 200 chars): {training_data[0]['input'][:200]}...")
    print(f"Output (first 200 chars): {training_data[0]['output'][:200]}...")
    print("=" * 80)
    
    print()
    print("🎉 DONE! Ready for fine-tuning on Kaggle/Colab")
    print(f"📁 Upload {output_file} to Kaggle as a dataset")

if __name__ == "__main__":
    main()
