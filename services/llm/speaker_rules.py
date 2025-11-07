# services/llm/speaker_rules.py
"""
Speaker Role Identification Rules
Uses linguistic patterns to identify doctor vs patient roles
Target: +2-3% accuracy improvement
"""

import re
from typing import Dict, List, Tuple


class SpeakerIdentifier:
    """Rule-based speaker role identification"""
    
    # Strong doctor indicators (high confidence)
    DOCTOR_PATTERNS_STRONG = [
        # Commands/Instructions (Arabic)
        r"خذ|تناول|استخدم|استعمل",  # Take, use
        r"سأصف|سأعطيك|سأوصف",  # I will prescribe
        r"يجب عليك|ينبغي|من الضروري",  # You must, should
        r"سأفحص|دعني أفحص|سأستمع",  # Let me examine
        r"سأطلب تحليل|احتاج إلى أشعة",  # Need tests/X-rays
        
        # Commands/Instructions (English)
        r"\bI will prescribe\b|\bI'll prescribe\b",
        r"\bYou need to\b|\bYou should\b|\bYou must\b",
        r"\bLet me examine\b|\bLet me check\b",
        r"\bI need to run tests\b|\bWe need an X-ray\b",
        
        # Medical procedures
        r"سأحقن|سأعمل عملية|سأخيط",  # Inject, operate, suture
        r"سأحيلك إلى|راجع مختص",  # Refer you to specialist
        
        # Diagnostic questions (Arabic)
        r"منذ متى|من متى|كم يوم",  # Since when, how long
        r"أين يؤلمك|ما نوع الألم",  # Where hurts, what type of pain
        r"هل تعاني من|هل لديك تاريخ",  # Do you suffer, medical history
        r"هل تأخذ أدوية|هل لديك حساسية",  # Medications, allergies
        
        # Diagnostic questions (English)
        r"\bHow long have you\b|\bSince when\b",
        r"\bWhere does it hurt\b|\bWhat kind of pain\b",
        r"\bAny history of\b|\bAny allergies\b",
        r"\bAre you taking any medications\b",
    ]
    
    # Moderate doctor indicators
    DOCTOR_PATTERNS_MODERATE = [
        # Medical terminology (Arabic)
        r"تشخيص|فحص سريري|علامات حيوية",  # Diagnosis, examination, vitals
        r"ضغط الدم|النبض|الحرارة",  # Blood pressure, pulse, temperature
        r"تحليل|أشعة|مختبر",  # Lab, X-ray, tests
        r"مضاد حيوي|مسكن|دواء",  # Antibiotic, painkiller, medicine
        r"جرعة|كل ست ساعات|مرتين يومياً",  # Dosage, frequency
        
        # Medical terminology (English)
        r"\bdiagnosis\b|\bexamination\b|\bvitals\b",
        r"\bblood pressure\b|\bpulse\b|\btemperature\b",
        r"\blaboratory\b|\bX-ray\b|\btests\b",
        r"\bantibiotic\b|\bpainkiller\b|\bmedicine\b",
        r"\bdosage\b|\bevery 6 hours\b|\btwice daily\b",
        
        # Professional language
        r"نتائج الفحص|التقرير الطبي",  # Examination results, medical report
        r"سأطلب استشارة|أنصحك بـ",  # Consultation, I advise you
    ]
    
    # Strong patient indicators
    PATIENT_PATTERNS_STRONG = [
        # Pain descriptions (Arabic)
        r"يؤلمني|أشعر بألم|عندي وجع",  # It hurts, I feel pain
        r"تعبان|مريض|مش قادر",  # Sick, can't (Egyptian)
        r"يعورني|تعبان|مو قادر",  # Hurts, sick (Gulf)
        
        # Pain descriptions (English)
        r"\bI feel pain\b|\bit hurts\b|\bI'm in pain\b",
        r"\bI can't\b|\bI'm unable to\b",
        
        # Symptom descriptions (Arabic)
        r"عندي|لدي|أعاني من",  # I have, I suffer from
        r"صداع|حمى|سعال|غثيان",  # Headache, fever, cough, nausea
        r"منذ يومين|من أمس|من أسبوع",  # Since 2 days, yesterday, week
        
        # Symptom descriptions (English)
        r"\bI have\b|\bI've been having\b|\bI suffer from\b",
        r"\bheadache\b|\bfever\b|\bcough\b|\bnausea\b",
        r"\bsince yesterday\b|\bfor 2 days\b|\bfor a week\b",
        
        # Personal concerns
        r"قلقان|خايف|متضايق",  # Worried, scared, upset
        r"ممكن|هل ممكن|ياريت",  # Can I, could you (requests)
    ]
    
    # Moderate patient indicators
    PATIENT_PATTERNS_MODERATE = [
        # Simple responses
        r"^نعم$|^لا$|^آه$|^أيوه$",  # Yes, no (Arabic)
        r"^\byes\b$|^\bno\b$|^\byeah\b$",  # Yes, no (English)
        
        # Short confirmations
        r"ماشي|حاضر|تمام|أوكي",  # OK, got it (Arabic)
        r"\bok\b|\bokay\b|\balright\b|\bgot it\b",  # OK (English)
        
        # Questions about treatment
        r"كم مرة|متى آخذ|لمدة كم",  # How many times, when to take
        r"هل فيه أعراض جانبية",  # Side effects
    ]
    
    # Questions patterns (usually doctor asks)
    QUESTION_WORDS_DOCTOR = [
        r"كيف|متى|أين|لماذا|ماذا|هل",  # How, when, where, why, what (Arabic)
        r"\bhow\b|\bwhen\b|\bwhere\b|\bwhy\b|\bwhat\b|\bdo you\b|\bdoes it\b",  # English
    ]
    
    # Short answers (usually patient responds)
    SHORT_ANSWER_THRESHOLD = 10  # Words
    
    def __init__(self):
        self.doctor_strong_re = [re.compile(p, re.IGNORECASE) for p in self.DOCTOR_PATTERNS_STRONG]
        self.doctor_moderate_re = [re.compile(p, re.IGNORECASE) for p in self.DOCTOR_PATTERNS_MODERATE]
        self.patient_strong_re = [re.compile(p, re.IGNORECASE) for p in self.PATIENT_PATTERNS_STRONG]
        self.patient_moderate_re = [re.compile(p, re.IGNORECASE) for p in self.PATIENT_PATTERNS_MODERATE]
        self.question_re = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_WORDS_DOCTOR]
    
    def analyze_utterance(self, text: str) -> Dict[str, float]:
        """
        Analyze a single utterance and return role scores
        Returns: {
            "doctor_score": float (0-1),
            "patient_score": float (0-1),
            "confidence": float (0-1)
        }
        """
        doctor_score = 0.0
        patient_score = 0.0
        
        # Count strong doctor patterns (weight: 0.3 each)
        for pattern in self.doctor_strong_re:
            if pattern.search(text):
                doctor_score += 0.3
        
        # Count moderate doctor patterns (weight: 0.15 each)
        for pattern in self.doctor_moderate_re:
            if pattern.search(text):
                doctor_score += 0.15
        
        # Count strong patient patterns (weight: 0.3 each)
        for pattern in self.patient_strong_re:
            if pattern.search(text):
                patient_score += 0.3
        
        # Count moderate patient patterns (weight: 0.15 each)
        for pattern in self.patient_moderate_re:
            if pattern.search(text):
                patient_score += 0.15
        
        # Question patterns (usually doctor)
        question_count = sum(1 for p in self.question_re if p.search(text))
        if question_count > 0:
            doctor_score += 0.2 * min(question_count, 2)
        
        # Short answer heuristic (usually patient)
        word_count = len(text.split())
        if word_count <= self.SHORT_ANSWER_THRESHOLD and patient_score == 0:
            # Short answers to questions are likely patient
            if question_count == 0:  # Not asking a question themselves
                patient_score += 0.1
        
        # Normalize scores to 0-1 range
        doctor_score = min(1.0, doctor_score)
        patient_score = min(1.0, patient_score)
        
        # Calculate confidence based on difference
        confidence = abs(doctor_score - patient_score)
        
        return {
            "doctor_score": doctor_score,
            "patient_score": patient_score,
            "confidence": confidence,
            "word_count": word_count
        }
    
    def identify_role(self, text: str) -> Tuple[str, float, str]:
        """
        Identify role from a single utterance
        Returns: (role, confidence, reasoning)
        """
        analysis = self.analyze_utterance(text)
        
        if analysis["doctor_score"] > analysis["patient_score"]:
            role = "Doctor"
            confidence = min(0.95, 0.5 + analysis["doctor_score"] * 0.5)
            reasoning = f"Medical terminology and diagnostic language (doctor score: {analysis['doctor_score']:.2f})"
        elif analysis["patient_score"] > analysis["doctor_score"]:
            role = "Patient"
            confidence = min(0.95, 0.5 + analysis["patient_score"] * 0.5)
            reasoning = f"Symptom descriptions and personal language (patient score: {analysis['patient_score']:.2f})"
        else:
            # Default to position-based (first speaker = doctor in medical context)
            role = "Unknown"
            confidence = 0.3
            reasoning = "Insufficient linguistic indicators"
        
        return role, confidence, reasoning
    
    def identify_conversation_roles(self, segments: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze entire conversation to identify all speaker roles
        Args:
            segments: [{"speaker": "SPEAKER_00", "text": "..."}]
        Returns:
            {"SPEAKER_00": {"role": "Doctor", "confidence": 0.85, "reasoning": "..."}}
        """
        speaker_analyses = {}
        
        # Analyze each segment
        for segment in segments:
            speaker_id = segment["speaker"]
            text = segment["text"]
            
            if speaker_id not in speaker_analyses:
                speaker_analyses[speaker_id] = {
                    "doctor_scores": [],
                    "patient_scores": [],
                    "utterances": []
                }
            
            analysis = self.analyze_utterance(text)
            speaker_analyses[speaker_id]["doctor_scores"].append(analysis["doctor_score"])
            speaker_analyses[speaker_id]["patient_scores"].append(analysis["patient_score"])
            speaker_analyses[speaker_id]["utterances"].append(text)
        
        # Aggregate scores for each speaker
        results = {}
        for speaker_id, data in speaker_analyses.items():
            avg_doctor = sum(data["doctor_scores"]) / len(data["doctor_scores"])
            avg_patient = sum(data["patient_scores"]) / len(data["patient_scores"])
            
            if avg_doctor > avg_patient:
                role = "Doctor"
                confidence = min(0.95, 0.5 + avg_doctor * 0.5)
                reasoning = f"Consistent medical terminology across {len(data['utterances'])} utterances (avg score: {avg_doctor:.2f})"
            elif avg_patient > avg_doctor:
                role = "Patient"
                confidence = min(0.95, 0.5 + avg_patient * 0.5)
                reasoning = f"Consistent symptom descriptions across {len(data['utterances'])} utterances (avg score: {avg_patient:.2f})"
            else:
                # Fallback: first speaker is usually doctor
                role = "Doctor" if speaker_id == "SPEAKER_00" else "Patient"
                confidence = 0.5
                reasoning = "Assigned by conversation position (first speaker assumed doctor)"
            
            results[speaker_id] = {
                "role": role,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        return results


if __name__ == "__main__":
    # Test speaker identification
    identifier = SpeakerIdentifier()
    
    test_segments = [
        {"speaker": "SPEAKER_00", "text": "أهلاً، ما الذي يؤلمك اليوم؟"},  # Doctor: question
        {"speaker": "SPEAKER_01", "text": "عندي صداع شديد منذ يومين"},  # Patient: symptom
        {"speaker": "SPEAKER_00", "text": "دعني أفحص ضغط الدم. هل تأخذ أدوية؟"},  # Doctor: examination
        {"speaker": "SPEAKER_01", "text": "لا، مافيش أدوية"},  # Patient: short answer
        {"speaker": "SPEAKER_00", "text": "سأصف لك مسكن، خذ حبة كل 6 ساعات"},  # Doctor: prescription
        {"speaker": "SPEAKER_01", "text": "تمام، شكراً يا دكتور"},  # Patient: confirmation
    ]
    
    print("Speaker Identification Test:")
    print("=" * 60)
    
    results = identifier.identify_conversation_roles(test_segments)
    
    for speaker_id, data in results.items():
        print(f"\n{speaker_id}:")
        print(f"  Role: {data['role']}")
        print(f"  Confidence: {data['confidence']:.2f}")
        print(f"  Reasoning: {data['reasoning']}")
    
    print("\n" + "=" * 60)
    print("Individual Utterance Analysis:")
    for seg in test_segments:
        role, conf, reason = identifier.identify_role(seg["text"])
        print(f"\n{seg['speaker']}: \"{seg['text'][:40]}...\"")
        print(f"  → {role} ({conf:.2f}): {reason}")
