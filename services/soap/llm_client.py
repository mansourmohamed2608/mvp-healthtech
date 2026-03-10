from typing import List, Dict, Any

import httpx


class LlmClient:
    def __init__(self, base_url: str, internal_secret: str, timeout_seconds: float = 200.0):
        self.base_url = base_url.rstrip("/")
        self.internal_secret = internal_secret
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
        session_id: str | None = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "sessionId": session_id,
        }
        headers = {"x-internal-secret": self.internal_secret}
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(f"{self.base_url}/generate", json=payload, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM generate failed: {resp.status_code}")
        data = resp.json()
        return data.get("text", "")
