"""
Vector-based RAG Integration with Gateway's VectorCacheService
Week 5 Day 31
"""
import httpx
from typing import List, Dict


class VectorRAG:
    """Client for vector-based retrieval using gateway's vector cache"""
    
    def __init__(self, gateway_url: str = "http://localhost:3000"):
        self.gateway_url = gateway_url
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def store_embedding(self, key: str, text: str, metadata: dict = None):
        """Store text with embedding in vector cache"""
        try:
            response = await self.client.post(
                f"{self.gateway_url}/rag/store",
                json={
                    "key": key,
                    "text": text,
                    "metadata": metadata or {},
                }
            )
            return response.json()
        except Exception as e:
            print(f"Failed to store embedding: {e}")
            return None
    
    async def search_similar(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for similar texts using vector similarity"""
        try:
            response = await self.client.post(
                f"{self.gateway_url}/rag/search",
                json={
                    "query": query,
                    "limit": limit,
                }
            )
            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except Exception as e:
            print(f"Failed to search embeddings: {e}")
            return []
    
    async def seed_medical_knowledge(self):
        """Seed vector store with medical knowledge"""
        medical_knowledge = [
            {
                "key": "hypertension_treatment",
                "text": "علاج ارتفاع ضغط الدم يشمل تغييرات نمط الحياة مثل تقليل الملح وممارسة الرياضة، بالإضافة إلى الأدوية مثل مثبطات الإنزيم المحول للأنجيوتنسين",
                "metadata": {"type": "treatment", "condition": "hypertension"},
            },
            {
                "key": "diabetes_symptoms",
                "text": "أعراض السكري تشمل العطش الشديد، كثرة التبول، التعب، وفقدان الوزن غير المبرر",
                "metadata": {"type": "symptoms", "condition": "diabetes"},
            },
            {
                "key": "headache_red_flags",
                "text": "العلامات الخطيرة للصداع: صداع مفاجئ وشديد، صداع مع حمى وتيبس الرقبة، صداع بعد إصابة الرأس، تغير في نمط الصداع المعتاد",
                "metadata": {"type": "red_flags", "condition": "headache"},
            },
        ]
        
        for item in medical_knowledge:
            await self.store_embedding(item["key"], item["text"], item["metadata"])
        
        print(f"Seeded {len(medical_knowledge)} medical knowledge items")


# Global instance
vector_rag = VectorRAG()
