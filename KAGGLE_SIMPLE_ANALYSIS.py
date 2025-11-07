"""
Simple Kaggle Medical Analysis
===============================
This version does ONLY what the model can actually do:
- Answer medical questions about the conversation
- Extract key medical facts

NOT trying to correct text or generate SOAP (use local services for that!)
"""

import os
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Your conversation (plain text)
CONVERSATION = """
السلام عليكم يا دكتور. وعليكم السلام. ازايك عاملة ايه? والله مش حاسة اني كويسة. اللسة عندي بقت حمرى ومتهيجة. ولما بخسل سناني اللسة بتنزف.
"""

# Questions to ask about the conversation
QUESTIONS = [
    "What is the main medical complaint?",
    "What symptoms does the patient describe?",
    "What diagnosis is mentioned?",
    "What treatment is recommended?",
]

# ============================================================================
# SETUP
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKING_DIR = "/kaggle/working"
MODEL_DIR = "/kaggle/input/medllm/models--Henrychur--MMed-Llama-3-8B"
MODEL_CACHE = os.path.join(WORKING_DIR, "models")

os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE
os.environ['HF_HOME'] = MODEL_CACHE

print("=" * 80)
print("SIMPLE MEDICAL ANALYSIS")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ============================================================================
# LOAD MODEL
# ============================================================================

HF_REPO_ID = "Henrychur/MMed-Llama-3-8B"

def _find_config_root(root_dir: str | Path) -> Path | None:
    root = Path(root_dir)
    if not root.exists():
        return None
    if (root / "config.json").exists():
        return root
    for cfg in (root / "snapshots").glob("*/config.json"):
        return cfg.parent
    return None

def _resolve_model_source(local_dir: str) -> tuple[str, dict]:
    cfg_root = _find_config_root(local_dir)
    if cfg_root is not None:
        print(f"✅ Found local config: {cfg_root.name}")
        return str(cfg_root), {"local_files_only": True}
    return HF_REPO_ID, {"local_files_only": False}

def load_model():
    print("Loading model...")
    model_path, extra = _resolve_model_source(MODEL_DIR)
    
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, **extra)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_cfg,
        device_map="auto",
        **extra,
    )
    
    print("✅ Model loaded\n")
    return model, tokenizer

# ============================================================================
# ANALYZE
# ============================================================================

def analyze_conversation(conversation, questions, model, tokenizer):
    """Ask specific questions about the medical conversation"""
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {question}")
        print("-" * 40)
        
        # Simple Q&A prompt (what the model is actually designed for)
        prompt = f"""Context: {conversation}

Question: {question}

Answer:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        print(f"Answer: {answer}\n")
        
        results.append({
            "question": question,
            "answer": answer
        })
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    model, tokenizer = load_model()
    
    print("=" * 80)
    print("ANALYZING CONVERSATION")
    print("=" * 80)
    print(f"Conversation: {CONVERSATION[:100]}...")
    print(f"Questions: {len(QUESTIONS)}")
    
    start = time.time()
    results = analyze_conversation(CONVERSATION, QUESTIONS, model, tokenizer)
    elapsed = time.time() - start
    
    # Save results
    output = {
        "conversation": CONVERSATION,
        "analysis": results,
        "processing_time": elapsed
    }
    
    output_file = os.path.join(WORKING_DIR, "simple_analysis.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"✅ Time: {elapsed:.1f}s")
    print(f"✅ Saved: {output_file}")

if __name__ == "__main__":
    main()
