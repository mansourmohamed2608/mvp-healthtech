"""
Quality Evaluation Script
Week 5 Day 33 (Oct 27, 2025)
Comprehensive evaluation of ASR WER, intent accuracy, and SOAP note quality
"""
import json
import os
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
from jiwer import wer, cer
import httpx


class QualityEvaluator:
    """Comprehensive quality evaluation for the HealthTech MVP"""
    
    def __init__(
        self,
        asr_url: str = "http://localhost:5000",
        llm_url: str = "http://localhost:5001",
        soap_url: str = "http://localhost:5003",
    ):
        self.asr_url = asr_url
        self.llm_url = llm_url
        self.soap_url = soap_url
        self.client = httpx.Client(timeout=60.0)
    
    def evaluate_asr_wer(self, golden_set_path: str) -> Dict:
        """
        Evaluate Word Error Rate (WER) on golden test set
        
        Expected format: golden_set.json
        [
          {
            "audio_path": "path/to/audio.wav",
            "reference": "النص المرجعي الصحيح",
            "dialect": "egyptian"
          }
        ]
        """
        print("\n" + "="*60)
        print("ASR WER EVALUATION")
        print("="*60)
        
        with open(golden_set_path, 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
        
        results = []
        references = []
        hypotheses = []
        
        for idx, item in enumerate(golden_set):
            print(f"\nProcessing {idx+1}/{len(golden_set)}: {item['audio_path']}")
            
            # Read and encode audio
            with open(item['audio_path'], 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            import base64
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Transcribe
            try:
                response = self.client.post(
                    f"{self.asr_url}/transcribe",
                    json={
                        "audio": audio_base64,
                        "dialect": item.get("dialect"),
                    }
                )
                hypothesis = response.json()["text"]
            except Exception as e:
                print(f"  Error: {e}")
                hypothesis = ""
            
            reference = item["reference"]
            references.append(reference)
            hypotheses.append(hypothesis)
            
            # Calculate WER for this sample
            sample_wer = wer(reference, hypothesis) * 100
            sample_cer = cer(reference, hypothesis) * 100
            
            results.append({
                "audio": item["audio_path"],
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": sample_wer,
                "cer": sample_cer,
                "dialect": item.get("dialect", "unknown"),
            })
            
            print(f"  WER: {sample_wer:.2f}%")
            print(f"  REF: {reference[:50]}...")
            print(f"  HYP: {hypothesis[:50]}...")
        
        # Overall metrics
        overall_wer = wer(references, hypotheses) * 100
        overall_cer = cer(references, hypotheses) * 100
        
        # Per-dialect metrics
        dialect_metrics = {}
        for dialect in set(item.get("dialect", "unknown") for item in results):
            dialect_refs = [r["reference"] for r in results if r.get("dialect") == dialect]
            dialect_hyps = [r["hypothesis"] for r in results if r.get("dialect") == dialect]
            if dialect_refs:
                dialect_metrics[dialect] = {
                    "wer": wer(dialect_refs, dialect_hyps) * 100,
                    "cer": cer(dialect_refs, dialect_hyps) * 100,
                    "count": len(dialect_refs),
                }
        
        print(f"\n{'='*60}")
        print(f"OVERALL WER: {overall_wer:.2f}%")
        print(f"OVERALL CER: {overall_cer:.2f}%")
        print(f"{'='*60}")
        
        for dialect, metrics in dialect_metrics.items():
            print(f"\n{dialect.upper()} ({metrics['count']} samples):")
            print(f"  WER: {metrics['wer']:.2f}%")
            print(f"  CER: {metrics['cer']:.2f}%")
        
        return {
            "overall_wer": overall_wer,
            "overall_cer": overall_cer,
            "dialect_metrics": dialect_metrics,
            "samples": results,
            "target_wer": 15.0,
            "passed": overall_wer < 15.0,
        }
    
    def evaluate_intent_accuracy(self, test_set_path: str) -> Dict:
        """
        Evaluate intent classification accuracy
        
        Expected format: intent_test.json
        [
          {
            "message": "أريد حجز موعد",
            "expected_intent": "appointment"
          }
        ]
        """
        print("\n" + "="*60)
        print("INTENT CLASSIFICATION EVALUATION")
        print("="*60)
        
        with open(test_set_path, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
        
        correct = 0
        total = len(test_set)
        results = []
        
        for idx, item in enumerate(test_set):
            print(f"\nProcessing {idx+1}/{total}")
            
            try:
                response = self.client.post(
                    f"{self.llm_url}/infer",
                    json={
                        "message": item["message"],
                        "sessionId": "eval-session",
                    }
                )
                predicted_intent = response.json()["intent"]
            except Exception as e:
                print(f"  Error: {e}")
                predicted_intent = "unknown"
            
            expected_intent = item["expected_intent"]
            is_correct = predicted_intent == expected_intent
            
            if is_correct:
                correct += 1
            
            results.append({
                "message": item["message"],
                "expected": expected_intent,
                "predicted": predicted_intent,
                "correct": is_correct,
            })
            
            print(f"  Message: {item['message'][:40]}...")
            print(f"  Expected: {expected_intent}, Predicted: {predicted_intent} {'✅' if is_correct else '❌'}")
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"INTENT ACCURACY: {accuracy:.2f}% ({correct}/{total})")
        print(f"{'='*60}")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
            "target_accuracy": 70.0,
            "passed": accuracy >= 70.0,
        }
    
    def evaluate_soap_quality(self, test_set_path: str) -> Dict:
        """
        Evaluate SOAP note generation quality
        
        Expected format: soap_test.json
        [
          {
            "transcript": "المريض يشكو من صداع...",
            "expected_diagnoses": ["صداع توتري"],
            "expected_medications": ["باراسيتامول"]
          }
        ]
        """
        print("\n" + "="*60)
        print("SOAP NOTE QUALITY EVALUATION")
        print("="*60)
        
        with open(test_set_path, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
        
        results = []
        completeness_scores = []
        factuality_scores = []
        
        for idx, item in enumerate(test_set):
            print(f"\nProcessing {idx+1}/{len(test_set)}")
            
            try:
                response = self.client.post(
                    f"{self.soap_url}/generate",
                    json={"transcript": item["transcript"]}
                )
                soap_note = response.json()["soap_note"]
            except Exception as e:
                print(f"  Error: {e}")
                soap_note = ""
            
            # Check completeness (all SOAP sections present)
            has_subjective = "الذاتي" in soap_note or "Subjective" in soap_note
            has_objective = "الموضوعي" in soap_note or "Objective" in soap_note
            has_assessment = "التقييم" in soap_note or "Assessment" in soap_note
            has_plan = "الخطة" in soap_note or "Plan" in soap_note
            
            completeness = sum([has_subjective, has_objective, has_assessment, has_plan]) / 4 * 100
            completeness_scores.append(completeness)
            
            # Check factuality (mentions expected items)
            factuality_correct = 0
            factuality_total = 0
            
            if "expected_diagnoses" in item:
                factuality_total += len(item["expected_diagnoses"])
                for diagnosis in item["expected_diagnoses"]:
                    if diagnosis in soap_note:
                        factuality_correct += 1
            
            if "expected_medications" in item:
                factuality_total += len(item["expected_medications"])
                for med in item["expected_medications"]:
                    if med in soap_note:
                        factuality_correct += 1
            
            factuality = (factuality_correct / factuality_total * 100) if factuality_total > 0 else 0
            factuality_scores.append(factuality)
            
            results.append({
                "transcript": item["transcript"][:50] + "...",
                "soap_note": soap_note,
                "completeness": completeness,
                "factuality": factuality,
                "has_all_sections": completeness == 100,
            })
            
            print(f"  Completeness: {completeness:.0f}%")
            print(f"  Factuality: {factuality:.0f}%")
        
        avg_completeness = np.mean(completeness_scores) if completeness_scores else 0
        avg_factuality = np.mean(factuality_scores) if factuality_scores else 0
        
        print(f"\n{'='*60}")
        print(f"AVG COMPLETENESS: {avg_completeness:.2f}%")
        print(f"AVG FACTUALITY: {avg_factuality:.2f}%")
        print(f"{'='*60}")
        
        return {
            "avg_completeness": avg_completeness,
            "avg_factuality": avg_factuality,
            "results": results,
            "target_completeness": 85.0,
            "target_factuality": 70.0,
            "passed": avg_completeness >= 85.0 and avg_factuality >= 70.0,
        }
    
    def generate_report(self, output_path: str = "quality_report.json"):
        """Generate comprehensive quality report"""
        print("\n" + "="*60)
        print("GENERATING QUALITY REPORT")
        print("="*60)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "evaluations": {},
            "summary": {},
        }
        
        # Run all evaluations
        eval_configs = [
            ("asr_wer", "data/golden_set.json", self.evaluate_asr_wer),
            ("intent_accuracy", "data/intent_test.json", self.evaluate_intent_accuracy),
            ("soap_quality", "data/soap_test.json", self.evaluate_soap_quality),
        ]
        
        all_passed = True
        for eval_name, test_file, eval_func in eval_configs:
            if os.path.exists(test_file):
                try:
                    result = eval_func(test_file)
                    report["evaluations"][eval_name] = result
                    if not result.get("passed", False):
                        all_passed = False
                except Exception as e:
                    print(f"\nError in {eval_name}: {e}")
                    report["evaluations"][eval_name] = {"error": str(e), "passed": False}
                    all_passed = False
            else:
                print(f"\nSkipping {eval_name}: {test_file} not found")
                report["evaluations"][eval_name] = {"skipped": True, "reason": "test file not found"}
        
        # Summary
        report["summary"]["all_tests_passed"] = all_passed
        report["summary"]["production_ready"] = all_passed
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"REPORT SAVED: {output_path}")
        print(f"PRODUCTION READY: {'✅ YES' if all_passed else '❌ NO'}")
        print(f"{'='*60}\n")
        
        return report


def main():
    evaluator = QualityEvaluator()
    report = evaluator.generate_report("quality_report.json")
    
    # Print summary
    print("\nQUALITY SUMMARY:")
    for eval_name, result in report["evaluations"].items():
        if "passed" in result:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"  {eval_name}: {status}")
        elif "skipped" in result:
            print(f"  {eval_name}: ⚠️  SKIPPED")
    
    return report


if __name__ == "__main__":
    main()
