#!/usr/bin/env python3
"""
FHIR Outbox Worker - Processes pending FHIR writes with retry logic
PR-8: FHIR writeback integrity

Run with: python services/fhir/outbox_worker.py
Or as cron: */5 * * * * python /app/services/fhir/outbox_worker.py

Environment:
- DATABASE_URL: PostgreSQL connection string
- FHIR_BASE_URL: FHIR server base URL
- FHIR_BEARER_TOKEN: Bearer token for FHIR auth
- INTERNAL_SECRET: For internal service auth
- OUTBOX_BATCH_SIZE: Max items per run (default: 50)
- OUTBOX_MAX_RETRIES: Max retry attempts (default: 3)
"""
import os
import sys
import json
import time
import logging
import httpx
import psycopg2
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fhir_outbox_worker")

DATABASE_URL = os.getenv("DATABASE_URL")
FHIR_BASE_URL = os.getenv("FHIR_BASE_URL")
FHIR_BEARER_TOKEN = os.getenv("FHIR_BEARER_TOKEN", "")
INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")
BATCH_SIZE = int(os.getenv("OUTBOX_BATCH_SIZE", "50"))
MAX_RETRIES = int(os.getenv("OUTBOX_MAX_RETRIES", "3"))

# Exponential backoff: 1min, 5min, 30min
RETRY_DELAYS_MINUTES = [1, 5, 30]


def get_db_connection():
    """Create database connection."""
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)


def get_pending_items(conn, limit: int = BATCH_SIZE) -> list:
    """Fetch pending outbox items ready for processing."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, tenant_id, soap_note_id, idempotency_key, payload,
                   attempts, max_attempts
            FROM fhir_outbox
            WHERE status IN ('pending', 'failed')
              AND (next_retry_at IS NULL OR next_retry_at <= NOW())
              AND attempts < max_attempts
            ORDER BY created_at ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        """, (limit,))
        return cur.fetchall()


def mark_processing(conn, outbox_id: str):
    """Mark item as processing."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE fhir_outbox
            SET status = 'processing', attempts = attempts + 1
            WHERE id = %s
        """, (outbox_id,))
    conn.commit()


def mark_success(conn, outbox_id: str, soap_note_id: str, fhir_resource_id: Optional[str] = None):
    """Mark item as successfully processed."""
    with conn.cursor() as cur:
        # Update outbox
        cur.execute("""
            UPDATE fhir_outbox
            SET status = 'success', processed_at = NOW()
            WHERE id = %s
        """, (outbox_id,))
        
        # Update soap_notes
        cur.execute("""
            UPDATE soap_notes
            SET fhir_status = 'success',
                fhir_written_at = NOW(),
                fhir_resource_id = COALESCE(%s, fhir_resource_id)
            WHERE id = %s
        """, (fhir_resource_id, soap_note_id))
    conn.commit()


def mark_failed(conn, outbox_id: str, soap_note_id: str, error: str, attempts: int, max_attempts: int):
    """Mark item as failed, schedule retry or dead letter."""
    with conn.cursor() as cur:
        if attempts >= max_attempts:
            # Dead letter - no more retries
            cur.execute("""
                UPDATE fhir_outbox
                SET status = 'dead_letter', last_error = %s
                WHERE id = %s
            """, (error, outbox_id))
            cur.execute("""
                UPDATE soap_notes
                SET fhir_status = 'failed', fhir_last_error = %s
                WHERE id = %s
            """, (error, soap_note_id))
        else:
            # Schedule retry with exponential backoff
            delay_idx = min(attempts - 1, len(RETRY_DELAYS_MINUTES) - 1)
            delay_minutes = RETRY_DELAYS_MINUTES[delay_idx]
            next_retry = datetime.utcnow() + timedelta(minutes=delay_minutes)
            
            cur.execute("""
                UPDATE fhir_outbox
                SET status = 'failed', last_error = %s, next_retry_at = %s
                WHERE id = %s
            """, (error, next_retry, outbox_id))
            cur.execute("""
                UPDATE soap_notes
                SET fhir_status = 'pending', fhir_attempts = %s, fhir_last_error = %s
                WHERE id = %s
            """, (attempts, error, soap_note_id))
    conn.commit()


def send_to_fhir(payload: Dict[str, Any], idempotency_key: str) -> Dict[str, Any]:
    """Send SOAP note to FHIR server."""
    if not FHIR_BASE_URL:
        raise RuntimeError("FHIR_BASE_URL not set")
    
    headers = {
        "Content-Type": "application/json",
        "Idempotency-Key": idempotency_key,
        "x-internal-secret": INTERNAL_SECRET,
    }
    if FHIR_BEARER_TOKEN:
        headers["Authorization"] = f"Bearer {FHIR_BEARER_TOKEN}"
    
    # Call the FHIR write endpoint
    fhir_service_url = os.getenv("FHIR_SERVICE_URL", "http://localhost:5004")
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{fhir_service_url}/write",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


def process_item(conn, item: tuple) -> bool:
    """Process a single outbox item."""
    outbox_id, tenant_id, soap_note_id, idempotency_key, payload, attempts, max_attempts = item
    
    logger.info(f"Processing outbox item {outbox_id} (attempt {attempts + 1}/{max_attempts})")
    
    try:
        mark_processing(conn, outbox_id)
        
        # Parse payload
        if isinstance(payload, str):
            payload = json.loads(payload)
        
        # Send to FHIR
        result = send_to_fhir(payload, idempotency_key)
        fhir_resource_id = result.get("resourceId") or result.get("id")
        
        mark_success(conn, outbox_id, soap_note_id, fhir_resource_id)
        logger.info(f"Successfully processed outbox item {outbox_id}")
        return True
        
    except Exception as e:
        error_msg = str(e)[:500]  # Truncate for DB
        logger.error(f"Failed to process outbox item {outbox_id}: {error_msg}")
        mark_failed(conn, outbox_id, soap_note_id, error_msg, attempts + 1, max_attempts)
        return False


def run_worker():
    """Main worker loop - process pending items."""
    logger.info("Starting FHIR outbox worker...")
    
    conn = get_db_connection()
    try:
        items = get_pending_items(conn, BATCH_SIZE)
        
        if not items:
            logger.info("No pending items to process")
            return 0
        
        logger.info(f"Found {len(items)} items to process")
        
        success_count = 0
        for item in items:
            if process_item(conn, item):
                success_count += 1
            # Small delay between items
            time.sleep(0.5)
        
        logger.info(f"Processed {success_count}/{len(items)} items successfully")
        return len(items) - success_count  # Return failure count
        
    finally:
        conn.close()


def get_stats(conn) -> Dict[str, int]:
    """Get outbox statistics."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT status, COUNT(*) as count
            FROM fhir_outbox
            GROUP BY status
        """)
        stats = {row[0]: row[1] for row in cur.fetchall()}
    return stats


def reconcile():
    """
    Daily reconciliation: find approved soap_notes with fhir_status='pending' but no outbox entry.
    Re-enqueue them to ensure eventual delivery.
    """
    logger.info("Starting FHIR reconciliation...")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Find orphaned notes (approved but stuck in pending with no outbox)
            cur.execute("""
                SELECT s.id, s.tenant_id, s.fhir_idempotency_key,
                       s.subjective, s.objective, s.assessment, s.plan,
                       s.patient_id, s.clinician_id, s.encounter_id, s.session_id
                FROM soap_notes s
                LEFT JOIN fhir_outbox o ON s.id = o.soap_note_id AND o.status IN ('pending', 'processing')
                WHERE s.approval_status = 'approved'
                  AND s.fhir_status = 'pending'
                  AND o.id IS NULL
                  AND s.updated_at < NOW() - INTERVAL '10 minutes'
                LIMIT 100
            """)
            orphaned = cur.fetchall()
        
        re_enqueued = 0
        for row in orphaned:
            (note_id, tenant_id, existing_key, subjective, objective, assessment, plan,
             patient_id, clinician_id, encounter_id, session_id) = row
            
            idempotency_key = existing_key or str(uuid.uuid4())
            payload = {
                "soapNote": {
                    "subjective": subjective or '',
                    "objective": objective or '',
                    "assessment": assessment or '',
                    "plan": plan or '',
                },
                "patientId": patient_id or '',
                "practitionerId": clinician_id or '',
                "encounterId": encounter_id or '',
                "sessionId": session_id or str(note_id),
            }
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO fhir_outbox (tenant_id, soap_note_id, idempotency_key, payload, status, next_retry_at)
                    VALUES (%s, %s, %s, %s, 'pending', NOW())
                    ON CONFLICT (idempotency_key) DO NOTHING
                """, (tenant_id, note_id, idempotency_key, json.dumps(payload)))
                if cur.rowcount > 0:
                    re_enqueued += 1
            conn.commit()
        
        # Record reconciliation run
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fhir_reconciliation (run_at, orphaned_found, re_enqueued, status)
                VALUES (NOW(), %s, %s, 'completed')
            """, (len(orphaned), re_enqueued))
        conn.commit()
        
        logger.info(f"Reconciliation complete: found={len(orphaned)}, re_enqueued={re_enqueued}")
        return {"orphaned_found": len(orphaned), "re_enqueued": re_enqueued}
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        failures = run_worker()
        sys.exit(1 if failures > 0 else 0)
    except Exception as e:
        logger.exception(f"Worker failed: {e}")
        sys.exit(1)
