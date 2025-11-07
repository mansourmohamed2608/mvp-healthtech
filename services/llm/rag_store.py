"""
RAG Knowledge Store
Week 5 Day 31 (Oct 25, 2025)
Stores and retrieves few-shot examples and clinic-specific knowledge
"""
import json
from typing import List, Dict, Any
from pathlib import Path


class RAGStore:
    """In-memory store for medical knowledge and few-shot examples"""
    
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Few-shot examples for different intents
        self.few_shot_examples = {
            "appointment": [
                {
                    "user": "أريد حجز موعد مع طبيب الأطفال",
                    "assistant": "بالطبع، سأساعدك في حجز موعد مع طبيب الأطفال. هل لديك تفضيل للوقت؟",
                },
                {
                    "user": "متى أقرب موعد متاح؟",
                    "assistant": "أقرب موعد متاح هو يوم الأحد الساعة 10 صباحاً. هل يناسبك هذا الموعد؟",
                },
            ],
            "symptom": [
                {
                    "user": "عندي صداع شديد منذ يومين",
                    "assistant": "أنا آسف لسماع ذلك. هل الصداع مستمر أم يأتي ويذهب؟ وهل هناك أعراض أخرى مثل الغثيان أو الحساسية للضوء؟",
                },
                {
                    "user": "ابني عنده حمى ودرجة حرارته 38.5",
                    "assistant": "أفهم قلقك. كم عمر طفلك؟ ومتى بدأت الحمى؟ هل هناك أعراض أخرى مثل السعال أو التقيؤ؟",
                },
            ],
            "prescription": [
                {
                    "user": "هل يمكنني تجديد وصفتي الطبية؟",
                    "assistant": "بالطبع. ما هو اسم الدواء الذي تحتاج لتجديده؟ ومتى كانت آخر مرة حصلت فيه على الوصفة؟",
                },
            ],
            "medical_history": [
                {
                    "user": "لدي حساسية من البنسلين",
                    "assistant": "شكراً لإبلاغي بذلك. سأسجل حساسيتك من البنسلين في ملفك الطبي. هل لديك أي حساسيات أخرى من الأدوية؟",
                },
            ],
        }
        
        # Common medical FAQs (Arabic medical knowledge)
        self.medical_faqs = [
            {
                "question": "ما هي الجرعة الآمنة من الباراسيتامول؟",
                "answer": "الجرعة الآمنة للبالغين هي 500-1000 ملغ كل 4-6 ساعات، بحد أقصى 4000 ملغ يومياً. للأطفال تختلف الجرعة حسب الوزن.",
            },
            {
                "question": "متى يجب علي زيارة الطبيب للصداع؟",
                "answer": "يجب زيارة الطبيب إذا كان الصداع شديداً ومفاجئاً، أو مصحوباً بحمى أو تيبس الرقبة، أو إذا استمر أكثر من 3 أيام.",
            },
            {
                "question": "هل الحمى دائماً خطيرة؟",
                "answer": "لا، الحمى هي استجابة طبيعية للجسم ضد العدوى. لكن يجب مراجعة الطبيب إذا تجاوزت 39.4 درجة أو استمرت أكثر من 3 أيام.",
            },
        ]
        
        # Clinic-specific protocols
        self.clinic_protocols = {
            "emergency_triage": [
                "إذا كان المريض يعاني من ألم في الصدر أو صعوبة في التنفس، حول الاتصال فوراً إلى الطوارئ",
                "إذا كانت الحمى أعلى من 40 درجة مع تشنجات، اطلب سيارة إسعاف",
            ],
            "appointment_hours": "نحن نعمل من الأحد إلى الخميس، 8 صباحاً - 8 مساءً. السبت 9 صباحاً - 2 مساءً.",
            "insurance_accepted": ["تأمين تعاوني", "بوبا", "التعاونية", "ميدغلف"],
        }
        
        # Load additional knowledge from files if available
        self._load_knowledge_files()
    
    def _load_knowledge_files(self):
        """Load knowledge from JSON files in knowledge directory"""
        examples_file = self.knowledge_dir / "few_shot_examples.json"
        if examples_file.exists():
            with open(examples_file, "r", encoding="utf-8") as f:
                loaded_examples = json.load(f)
                self.few_shot_examples.update(loaded_examples)
        
        faqs_file = self.knowledge_dir / "medical_faqs.json"
        if faqs_file.exists():
            with open(faqs_file, "r", encoding="utf-8") as f:
                loaded_faqs = json.load(f)
                self.medical_faqs.extend(loaded_faqs)
    
    def get_few_shot_examples(self, intent: str, limit: int = 3) -> List[Dict[str, str]]:
        """Get few-shot examples for a specific intent"""
        examples = self.few_shot_examples.get(intent, [])
        return examples[:limit]
    
    def get_relevant_faqs(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        """Get relevant FAQs based on query (simple keyword matching)"""
        # Simple keyword matching (in production, use embeddings)
        query_lower = query.lower()
        scored_faqs = []
        
        for faq in self.medical_faqs:
            score = 0
            question_lower = faq["question"].lower()
            
            # Count matching words
            query_words = set(query_lower.split())
            question_words = set(question_lower.split())
            score = len(query_words & question_words)
            
            if score > 0:
                scored_faqs.append((score, faq))
        
        # Sort by score and return top results
        scored_faqs.sort(reverse=True, key=lambda x: x[0])
        return [faq for score, faq in scored_faqs[:limit]]
    
    def get_clinic_protocol(self, protocol_type: str) -> Any:
        """Get clinic-specific protocol"""
        return self.clinic_protocols.get(protocol_type, None)
    
    def add_few_shot_example(self, intent: str, user_text: str, assistant_text: str):
        """Add a new few-shot example"""
        if intent not in self.few_shot_examples:
            self.few_shot_examples[intent] = []
        
        self.few_shot_examples[intent].append({
            "user": user_text,
            "assistant": assistant_text,
        })
        
        # Save to file
        self._save_few_shot_examples()
    
    def add_faq(self, question: str, answer: str):
        """Add a new FAQ"""
        self.medical_faqs.append({
            "question": question,
            "answer": answer,
        })
        
        # Save to file
        self._save_faqs()
    
    def _save_few_shot_examples(self):
        """Save few-shot examples to file"""
        examples_file = self.knowledge_dir / "few_shot_examples.json"
        with open(examples_file, "w", encoding="utf-8") as f:
            json.dump(self.few_shot_examples, f, ensure_ascii=False, indent=2)
    
    def _save_faqs(self):
        """Save FAQs to file"""
        faqs_file = self.knowledge_dir / "medical_faqs.json"
        with open(faqs_file, "w", encoding="utf-8") as f:
            json.dump(self.medical_faqs, f, ensure_ascii=False, indent=2)


# Global instance
rag_store = RAGStore()
