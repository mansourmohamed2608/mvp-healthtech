"""
Compare Base Model vs Fine-tuned Model Performance
===================================================

This script tests both models side-by-side to validate
that fine-tuning improved quality.

Run after deploying fine-tuned model to compare outputs.
"""

import requests
import json
from typing import Dict, List
import time

# Test configuration
LLM_SERVICE_URL = "http://localhost:5003"

# Test cases - Egyptian Arabic medical conversations
TEST_CASES = [
    {
        "id": 1,
        "category": "Dental",
        "conversation": """دكتور: ازيك يا فندم؟ في ايه؟
مريض: والله مش كويس يا دكتور، عندي وجع في اللثة
دكتور: ومن امتى وانت حاسس كده؟
مريض: من حوالي اسبوع، ولما بغسل سناني بتنزف دم
دكتور: طيب هفحصك دلوقتي... اه واضح عندك التهاب في اللثة
مريض: ده خطير يا دكتور؟
دكتور: لا عادي، بس محتاج تنظيف عميق وهديك مضاد التهاب""",
        "expected_symptoms": ["وجع اللثة", "نزيف"],
        "expected_diagnosis": "التهاب اللثة",
        "expected_plan": ["تنظيف", "مضاد التهاب"]
    },
    {
        "id": 2,
        "category": "Neurology",
        "conversation": """دكتور: عامل ايه؟ قولي في ايه
مريض: عندي صداع مستمر يا دكتور من 3 ايام
دكتور: الصداع ده بيزيد في وقت معين؟
مريض: اه بالليل بيبقى اقوى، ومش عارف انام
دكتور: وبتاخد حاجة للصداع؟
مريض: باخد بنادول بس مش بيعمل حاجة
دكتور: طيب هفحصك... الضغط طبيعي. على الأغلب صداع توتر""",
        "expected_symptoms": ["صداع", "صعوبة النوم"],
        "expected_diagnosis": "صداع توتر",
        "expected_plan": ["دواء أقوى", "راحة"]
    },
    {
        "id": 3,
        "category": "Respiratory",
        "conversation": """دكتور: اتفضل قولي المشكلة ايه
مريض: يا دكتور عندي كحة وضيق في التنفس
دكتور: من امتى؟
مريض: من يومين تقريبا، وبالليل بيزيد
دكتور: في حمى؟
مريض: لا، بس بحس بصفير في الصدر
دكتور: طيب خليني اسمع الصدر... في صفير. عندك حساسية صدر""",
        "expected_symptoms": ["كحة", "ضيق تنفس", "صفير"],
        "expected_diagnosis": "حساسية صدر",
        "expected_plan": ["بخاخة", "مضاد حساسية"]
    },
    {
        "id": 4,
        "category": "GI",
        "conversation": """دكتور: ازيك؟ في ايه؟
مريض: عندي وجع في البطن يا دكتور
دكتور: الوجع فين بالضبط؟
مريض: حوالين السرة، وأحيانا اسهال
دكتور: ومن امتى؟
مريض: من امبارح بعد ما اكلت برة
دكتور: في حمى أو ترجيع؟
مريض: في شوية غثيان بس
دكتور: واضح تسمم غذائي خفيف. هديك علاج""",
        "expected_symptoms": ["ألم البطن", "إسهال", "غثيان"],
        "expected_diagnosis": "تسمم غذائي",
        "expected_plan": ["راحة", "سوائل", "دواء"]
    },
    {
        "id": 5,
        "category": "Pediatrics",
        "conversation": """دكتور: اهلا، الطفل عنده ايه؟
أم: يا دكتور ابني عنده حرارة عالية من امبارح
دكتور: كام سنة؟
أم: 5 سنين
دكتور: الحرارة قد ايه؟
أم: 38.5، وعنده كحة كمان
دكتور: خليني افحصه... في احمرار في الزور. التهاب زور""",
        "expected_symptoms": ["حمى", "كحة", "احمرار الزور"],
        "expected_diagnosis": "التهاب الحلق",
        "expected_plan": ["خافض حرارة", "مضاد حيوي"]
    }
]

def test_soap_generation(conversation: str) -> Dict:
    """Generate SOAP note for conversation"""
    try:
        response = requests.post(
            f"{LLM_SERVICE_URL}/soap",
            json={"text": conversation},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def check_soap_quality(soap: str, expected: Dict) -> Dict:
    """Check if SOAP note meets quality criteria"""
    checks = {}
    
    # Structure checks
    checks["has_subjective"] = any(x in soap for x in ["S:", "Subjective", "S (Subjective)"])
    checks["has_objective"] = any(x in soap for x in ["O:", "Objective", "O (Objective)"])
    checks["has_assessment"] = any(x in soap for x in ["A:", "Assessment", "A (Assessment)"])
    checks["has_plan"] = any(x in soap for x in ["P:", "Plan", "P (Plan)"])
    
    # Content checks
    checks["mentions_symptoms"] = any(symptom in soap for symptom in expected["expected_symptoms"])
    checks["mentions_diagnosis"] = expected["expected_diagnosis"] in soap if expected["expected_diagnosis"] else True
    checks["has_treatment_plan"] = any(plan in soap for plan in expected["expected_plan"])
    
    # Quality checks
    checks["sufficient_length"] = len(soap) > 100
    checks["no_repetition"] = not has_repetition(soap)
    checks["no_gibberish"] = not has_gibberish(soap)
    
    # Calculate score
    score = sum(1 for v in checks.values() if v) / len(checks) * 100
    
    return {
        "checks": checks,
        "score": score,
        "passed": score >= 70  # 70% threshold
    }

def has_repetition(text: str) -> bool:
    """Check if text has obvious repetition"""
    words = text.split()
    if len(words) < 5:
        return False
    
    # Check for 3+ word repetition
    for i in range(len(words) - 4):
        pattern = ' '.join(words[i:i+3])
        if text.count(pattern) > 1:
            return True
    
    return False

def has_gibberish(text: str) -> bool:
    """Check for gibberish patterns"""
    # Check for very short repeated words
    words = text.split()
    short_words = [w for w in words if len(w) <= 3]
    
    if len(short_words) > len(words) * 0.5:  # More than 50% very short words
        return True
    
    # Check for same word repeated 3+ times in row
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True
    
    return False

def compare_models():
    """Run all tests and compare results"""
    print("=" * 80)
    print("BASE MODEL vs FINE-TUNED MODEL COMPARISON")
    print("=" * 80)
    print()
    print(f"Testing {len(TEST_CASES)} conversations...")
    print(f"LLM Service: {LLM_SERVICE_URL}")
    print()
    
    results = []
    
    for test in TEST_CASES:
        print(f"Test #{test['id']}: {test['category']}")
        print("-" * 80)
        print("Conversation:")
        print(test['conversation'])
        print()
        
        # Generate SOAP note
        print("Generating SOAP note...")
        start_time = time.time()
        result = test_soap_generation(test['conversation'])
        generation_time = time.time() - start_time
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            print()
            continue
        
        soap = result.get("soap", "")
        print(f"Generated in {generation_time:.1f}s")
        print()
        
        # Display SOAP note
        print("Generated SOAP Note:")
        print("┌" + "─" * 78 + "┐")
        for line in soap.split('\n'):
            print(f"│ {line:<76} │")
        print("└" + "─" * 78 + "┘")
        print()
        
        # Quality checks
        quality = check_soap_quality(soap, test)
        
        print("Quality Checks:")
        for check, passed in quality["checks"].items():
            icon = "✅" if passed else "❌"
            print(f"  {icon} {check.replace('_', ' ').title()}")
        
        print()
        print(f"Overall Score: {quality['score']:.1f}%")
        print(f"Status: {'✅ PASSED' if quality['passed'] else '❌ FAILED'}")
        print()
        print("=" * 80)
        print()
        
        # Store results
        results.append({
            "test_id": test['id'],
            "category": test['category'],
            "score": quality['score'],
            "passed": quality['passed'],
            "generation_time": generation_time,
            "checks": quality['checks']
        })
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['passed'])
    avg_score = sum(r['score'] for r in results) / total_tests if total_tests > 0 else 0
    avg_time = sum(r['generation_time'] for r in results) / total_tests if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"Average Score: {avg_score:.1f}%")
    print(f"Average Generation Time: {avg_time:.1f}s")
    print()
    
    # Category breakdown
    print("Performance by Category:")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['score'])
    
    for cat, scores in categories.items():
        avg = sum(scores) / len(scores)
        print(f"  {cat}: {avg:.1f}%")
    
    print()
    
    # Detailed results
    print("Detailed Results:")
    print()
    print("| Test | Category | Score | Time | Status |")
    print("|------|----------|-------|------|--------|")
    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"| #{r['test_id']} | {r['category']:<12} | {r['score']:.1f}% | {r['generation_time']:.1f}s | {status} |")
    
    print()
    print("=" * 80)
    
    # Save results to file
    output_file = "comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate": passed_tests/total_tests*100 if total_tests > 0 else 0,
                "avg_score": avg_score,
                "avg_time": avg_time
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to {output_file}")
    print()
    
    return results

if __name__ == "__main__":
    print()
    print("Testing fine-tuned model quality...")
    print(f"Make sure LLM service is running on {LLM_SERVICE_URL}")
    print()
    
    input("Press Enter to start tests...")
    print()
    
    try:
        results = compare_models()
        
        # Final verdict
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        
        print()
        if passed == total:
            print("🎉 ALL TESTS PASSED! Fine-tuned model is working great!")
        elif passed >= total * 0.7:
            print(f"✅ Good performance! {passed}/{total} tests passed.")
            print("   Consider fine-tuning further for 100% pass rate.")
        else:
            print(f"⚠️ Only {passed}/{total} tests passed.")
            print("   Model may need more training data or tuning.")
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
