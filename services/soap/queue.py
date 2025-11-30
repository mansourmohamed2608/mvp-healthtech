import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any
import aioredis
import httpx
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("soap-queue")

QUEUE_KEY = os.getenv("SOAP_QUEUE_KEY", "soap_jobs")
REDIS_URL = os.getenv("SOAP_QUEUE_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0")
SOAP_URL = os.getenv("SOAP_SERVICE_URL", "http://localhost:5003")
INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required for soap worker")
if not INTERNAL_SECRET:
    raise RuntimeError("INTERNAL_SECRET is required for soap worker")
if not SOAP_URL:
    raise RuntimeError("SOAP_SERVICE_URL is required for soap worker")
MAX_ATTEMPTS = int(os.getenv("SOAP_JOB_MAX_ATTEMPTS", "3"))
BACKOFF_MS = int(os.getenv("SOAP_JOB_BACKOFF_MS", "500"))


async def get_client():
    return await aioredis.from_url(REDIS_URL, decode_responses=True)


async def fetch_job(redis, job_id: str) -> Optional[Dict[str, Any]]:
    data = await redis.hgetall(job_id)
    if not data:
        return None
    return {k: json.loads(v) if k == "payload" else v for k, v in data.items()}


async def set_job(redis, job_id: str, job: Dict[str, Any]):
    # Serialize payload separately
    job_to_store = {k: (json.dumps(v) if k == "payload" else v) for k, v in job.items()}
    await redis.hset(job_id, mapping=job_to_store)


async def mark_status(redis, job_id: str, status: str, result: Optional[Dict[str, Any]] = None):
    updates = {"status": status}
    if result:
        updates["result"] = json.dumps(result)
    await redis.hset(job_id, mapping=updates)


async def update_db(pool: asyncpg.Pool, query: str, *args):
    async with pool.acquire() as conn:
        await conn.execute(query, *args)


async def worker_loop():
    redis = await get_client()
    db_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            job_id = await redis.brpoplpush(QUEUE_KEY, f"{QUEUE_KEY}:processing", timeout=5)
            if not job_id:
                continue

            job = await fetch_job(redis, job_id)
            if not job:
                await redis.lrem(f"{QUEUE_KEY}:processing", 0, job_id)
                continue

            attempts = int(job.get("attempts", 0))
            payload = job.get("payload", {})
            try:
                await update_db(
                    db_pool,
                    "UPDATE soap_jobs SET status='processing', attempts=$2, updated_at=now() WHERE job_id=$1",
                    job_id,
                    attempts,
                )
                resp = await client.post(
                    f"{SOAP_URL}/generate",
                    json=payload,
                    headers={"x-internal-secret": INTERNAL_SECRET, "x-correlation-id": job.get("correlationId")},
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"SOAP status {resp.status_code}")
                result = resp.json()
                note_id = result.get("id")
                await update_db(
                    db_pool,
                    "UPDATE soap_jobs SET status='done', note_id=$2, updated_at=now(), last_error=NULL WHERE job_id=$1",
                    job_id,
                    note_id,
                )
                await mark_status(redis, job_id, "done", result)
            except Exception as e:
                attempts += 1
                job["attempts"] = attempts
                await set_job(redis, job_id, job)
                err_msg = str(e)
                if attempts >= MAX_ATTEMPTS:
                    await update_db(
                        db_pool,
                        "UPDATE soap_jobs SET status='failed', attempts=$2, last_error=$3, updated_at=now() WHERE job_id=$1",
                        job_id,
                        attempts,
                        err_msg,
                    )
                    await mark_status(redis, job_id, "failed", {"error": err_msg})
                else:
                    await update_db(
                        db_pool,
                        "UPDATE soap_jobs SET status='pending', attempts=$2, last_error=$3, updated_at=now() WHERE job_id=$1",
                        job_id,
                        attempts,
                        err_msg,
                    )
                    await asyncio.sleep(BACKOFF_MS / 1000.0)
                    await redis.lpush(QUEUE_KEY, job_id)
            finally:
                await redis.lrem(f"{QUEUE_KEY}:processing", 0, job_id)


if __name__ == "__main__":
    asyncio.run(worker_loop())
