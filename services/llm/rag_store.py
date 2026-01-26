"""
RAG Knowledge Store
Week 5 Day 31 (Oct 25, 2025)
Stores and retrieves few-shot examples and clinic-specific knowledge
"""
import json
import logging
import math
import os
import re
import uuid
from typing import List, Dict, Any
from pathlib import Path

import httpx

logger = logging.getLogger("rag-store")


def _normalize_tenant_id(value: str | None) -> str:
    if not value:
        return "default"
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return normalized or "default"


def _simple_embedding(text: str, dim: int = 128) -> List[float]:
    vector = [0.0] * dim
    if not text:
        return vector
    for ch in text:
        vector[ord(ch) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


class QdrantStore:
    def __init__(self, url: str, collection: str, vector_size: int = 128, timeout: float = 2.0):
        self.url = url.rstrip("/")
        self.collection = collection
        self.vector_size = vector_size
        self.client = httpx.Client(timeout=timeout)
        self.ready = False
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            resp = self.client.get(f"{self.url}/collections/{self.collection}")
            if resp.status_code == 200:
                self.ready = True
                return
            if resp.status_code == 404:
                payload = {
                    "vectors": {"size": self.vector_size, "distance": "Cosine"},
                }
                created = self.client.put(
                    f"{self.url}/collections/{self.collection}",
                    json=payload,
                )
                if created.status_code in (200, 201):
                    self.ready = True
                    return
            logger.warning("Qdrant collection not ready", extra={"status": resp.status_code})
        except Exception as exc:
            logger.warning("Qdrant init failed", extra={"error": str(exc)})
        self.ready = False

    def upsert(self, vector: List[float], payload: Dict[str, Any]) -> str | None:
        if not self.ready:
            return None
        point_id = uuid.uuid4().hex
        body = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload,
                }
            ]
        }
        resp = self.client.put(
            f"{self.url}/collections/{self.collection}/points",
            json=body,
            params={"wait": "true"},
        )
        if resp.status_code not in (200, 201):
            logger.warning("Qdrant upsert failed", extra={"status": resp.status_code})
            return None
        return point_id

    def search(self, vector: List[float], tenant_id: str, item_type: str, limit: int) -> List[Dict[str, Any]]:
        if not self.ready:
            return []
        body = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
            "filter": {
                "must": [
                    {"key": "tenant_id", "match": {"value": tenant_id}},
                    {"key": "item_type", "match": {"value": item_type}},
                ]
            },
        }
        resp = self.client.post(
            f"{self.url}/collections/{self.collection}/points/search",
            json=body,
        )
        if resp.status_code != 200:
            logger.warning("Qdrant search failed", extra={"status": resp.status_code})
            return []
        data = resp.json()
        return data.get("result", []) if isinstance(data, dict) else []

    def scroll(self, tenant_id: str, item_type: str, limit: int) -> List[Dict[str, Any]]:
        if not self.ready:
            return []
        body = {
            "limit": limit,
            "with_payload": True,
            "filter": {
                "must": [
                    {"key": "tenant_id", "match": {"value": tenant_id}},
                    {"key": "item_type", "match": {"value": item_type}},
                ]
            },
        }
        resp = self.client.post(
            f"{self.url}/collections/{self.collection}/points/scroll",
            json=body,
        )
        if resp.status_code != 200:
            logger.warning("Qdrant scroll failed", extra={"status": resp.status_code})
            return []
        data = resp.json()
        return data.get("result", {}).get("points", []) if isinstance(data, dict) else []


class RAGStore:
    """In-memory store for medical knowledge and few-shot examples"""

    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "va_rag")
        self.qdrant = QdrantStore(qdrant_url, qdrant_collection) if qdrant_url else None

        self._loaded_tenants: set[str] = set()

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
        self.medical_faqs_by_tenant: Dict[str, List[Dict[str, str]]] = {
            "default": [
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
        }

        # Clinic-specific protocols
        self.clinic_protocols_by_tenant: Dict[str, Dict[str, Any]] = {
            "default": {
                "emergency_triage": [
                    "إذا كان المريض يعاني من ألم في الصدر أو صعوبة في التنفس، حول الاتصال فوراً إلى الطوارئ",
                    "إذا كانت الحمى أعلى من 40 درجة مع تشنجات، اطلب سيارة إسعاف",
                ],
                "appointment_hours": "نحن نعمل من الأحد إلى الخميس، 8 صباحاً - 8 مساءً. السبت 9 صباحاً - 2 مساءً.",
                "insurance_accepted": ["تأمين تعاوني", "بوبا", "التعاونية", "ميدغلف"],
            }
        }

        # Free-form clinic notes/policies
        self.clinic_notes_by_tenant: Dict[str, List[Dict[str, Any]]] = {"default": []}

        # Load additional knowledge from files if available
        self._load_knowledge_files()
    
    def _load_knowledge_files(self):
        """Load knowledge from JSON files in knowledge directory"""
        examples_file = self.knowledge_dir / "few_shot_examples.json"
        if examples_file.exists():
            with open(examples_file, "r", encoding="utf-8") as f:
                loaded_examples = json.load(f)
                self.few_shot_examples.update(loaded_examples)

        self._load_tenant_files("default")

    def _tenant_dir(self, tenant_id: str) -> Path:
        if tenant_id == "default":
            return self.knowledge_dir
        return self.knowledge_dir / "tenants" / tenant_id

    def _load_tenant_files(self, tenant_id: str) -> None:
        if tenant_id in self._loaded_tenants:
            return
        self._loaded_tenants.add(tenant_id)
        tenant_dir = self._tenant_dir(tenant_id)
        tenant_dir.mkdir(parents=True, exist_ok=True)

        faqs_file = tenant_dir / "medical_faqs.json"
        if faqs_file.exists():
            with open(faqs_file, "r", encoding="utf-8") as f:
                loaded_faqs = json.load(f)
                if isinstance(loaded_faqs, list):
                    self.medical_faqs_by_tenant.setdefault(tenant_id, []).extend(loaded_faqs)

        notes_file = tenant_dir / "clinic_notes.json"
        if notes_file.exists():
            with open(notes_file, "r", encoding="utf-8") as f:
                loaded_notes = json.load(f)
                if isinstance(loaded_notes, list):
                    self.clinic_notes_by_tenant.setdefault(tenant_id, []).extend(loaded_notes)

        protocols_file = tenant_dir / "clinic_protocols.json"
        if protocols_file.exists():
            with open(protocols_file, "r", encoding="utf-8") as f:
                loaded_protocols = json.load(f)
                if isinstance(loaded_protocols, dict):
                    self.clinic_protocols_by_tenant[tenant_id] = loaded_protocols
    
    def get_few_shot_examples(self, intent: str, limit: int = 3) -> List[Dict[str, str]]:
        """Get few-shot examples for a specific intent"""
        examples = self.few_shot_examples.get(intent, [])
        return examples[:limit]
    
    def get_relevant_faqs(self, query: str, limit: int = 3, tenant_id: str | None = None) -> List[Dict[str, str]]:
        """Get relevant FAQs based on query (simple keyword matching)"""
        tenant = _normalize_tenant_id(tenant_id)
        if self.qdrant and self.qdrant.ready:
            vector = _simple_embedding(query)
            results = self.qdrant.search(vector, tenant, "faq", limit)
            faqs: List[Dict[str, str]] = []
            for item in results:
                payload = item.get("payload") or {}
                question = payload.get("question") or ""
                answer = payload.get("answer") or ""
                if question or answer:
                    faqs.append({"question": question, "answer": answer})
            return faqs

        self._load_tenant_files(tenant)
        faqs = self.medical_faqs_by_tenant.get(tenant, self.medical_faqs_by_tenant.get("default", []))
        query_lower = query.lower()
        scored_faqs = []

        for faq in faqs:
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

    def get_relevant_notes(self, query: str, limit: int = 3, tenant_id: str | None = None) -> List[Dict[str, Any]]:
        """Get relevant clinic notes based on keyword matching"""
        tenant = _normalize_tenant_id(tenant_id)
        if self.qdrant and self.qdrant.ready:
            vector = _simple_embedding(query)
            results = self.qdrant.search(vector, tenant, "note", limit)
            notes: List[Dict[str, Any]] = []
            for item in results:
                payload = item.get("payload") or {}
                text = payload.get("text") or ""
                title = payload.get("title") or "معلومة"
                metadata = payload.get("metadata") or {}
                if text:
                    notes.append({"title": title, "text": text, "metadata": metadata})
            return notes

        self._load_tenant_files(tenant)
        notes = self.clinic_notes_by_tenant.get(tenant, self.clinic_notes_by_tenant.get("default", []))
        query_lower = query.lower()
        scored_notes = []
        for note in notes:
            text = (note.get("text") or "").lower()
            title = (note.get("title") or "").lower()
            score = 0
            query_words = set(query_lower.split())
            score += len(query_words & set(text.split()))
            score += len(query_words & set(title.split()))
            if score > 0:
                scored_notes.append((score, note))
        scored_notes.sort(reverse=True, key=lambda x: x[0])
        return [note for score, note in scored_notes[:limit]]
    
    def get_clinic_protocol(self, protocol_type: str, tenant_id: str | None = None) -> Any:
        """Get clinic-specific protocol"""
        tenant = _normalize_tenant_id(tenant_id)
        protocols = self.clinic_protocols_by_tenant.get(tenant) or self.clinic_protocols_by_tenant.get("default", {})
        return protocols.get(protocol_type, None)

    def get_protocols(self, tenant_id: str | None = None) -> Dict[str, Any]:
        tenant = _normalize_tenant_id(tenant_id)
        return self.clinic_protocols_by_tenant.get(tenant) or self.clinic_protocols_by_tenant.get("default", {})
    
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
    
    def add_faq(self, question: str, answer: str, tenant_id: str | None = None):
        """Add a new FAQ"""
        tenant = _normalize_tenant_id(tenant_id)
        entry = {"question": question, "answer": answer}
        if self.qdrant and self.qdrant.ready:
            vector = _simple_embedding(f"{question} {answer}")
            payload = {
                "tenant_id": tenant,
                "item_type": "faq",
                "question": question,
                "answer": answer,
            }
            self.qdrant.upsert(vector, payload)
            return

        self._load_tenant_files(tenant)
        self.medical_faqs_by_tenant.setdefault(tenant, []).append(entry)

        # Save to file
        self._save_faqs(tenant)

    def add_note(self, text: str, title: str | None = None, metadata: Dict[str, Any] | None = None, tenant_id: str | None = None):
        """Add a free-form clinic note/policy"""
        tenant = _normalize_tenant_id(tenant_id)
        entry = {
            "title": title or "معلومة",
            "text": text,
            "metadata": metadata or {},
        }
        if self.qdrant and self.qdrant.ready:
            vector = _simple_embedding(text)
            payload = {
                "tenant_id": tenant,
                "item_type": "note",
                "title": entry["title"],
                "text": entry["text"],
                "metadata": entry["metadata"],
            }
            self.qdrant.upsert(vector, payload)
            return

        self._load_tenant_files(tenant)
        self.clinic_notes_by_tenant.setdefault(tenant, []).append(entry)
        self._save_notes(tenant)

    def list_notes(self, tenant_id: str | None = None, limit: int = 50) -> List[Dict[str, Any]]:
        tenant = _normalize_tenant_id(tenant_id)
        if self.qdrant and self.qdrant.ready:
            points = self.qdrant.scroll(tenant, "note", limit)
            notes: List[Dict[str, Any]] = []
            for point in points:
                payload = point.get("payload") or {}
                text = payload.get("text") or ""
                if not text:
                    continue
                notes.append(
                    {
                        "title": payload.get("title") or "معلومة",
                        "text": text,
                        "metadata": payload.get("metadata") or {},
                    }
                )
            return notes

        self._load_tenant_files(tenant)
        notes = self.clinic_notes_by_tenant.get(tenant, [])
        return list(notes)[:limit]
    
    def _save_few_shot_examples(self):
        """Save few-shot examples to file"""
        examples_file = self.knowledge_dir / "few_shot_examples.json"
        with open(examples_file, "w", encoding="utf-8") as f:
            json.dump(self.few_shot_examples, f, ensure_ascii=False, indent=2)

    def _save_faqs(self, tenant_id: str):
        """Save FAQs to file"""
        tenant_dir = self._tenant_dir(tenant_id)
        tenant_dir.mkdir(parents=True, exist_ok=True)
        faqs_file = tenant_dir / "medical_faqs.json"
        faqs = self.medical_faqs_by_tenant.get(tenant_id, [])
        with open(faqs_file, "w", encoding="utf-8") as f:
            json.dump(faqs, f, ensure_ascii=False, indent=2)

    def _save_notes(self, tenant_id: str):
        """Save clinic notes to file"""
        tenant_dir = self._tenant_dir(tenant_id)
        tenant_dir.mkdir(parents=True, exist_ok=True)
        notes_file = tenant_dir / "clinic_notes.json"
        notes = self.clinic_notes_by_tenant.get(tenant_id, [])
        with open(notes_file, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)


# Global instance
rag_store = RAGStore()
