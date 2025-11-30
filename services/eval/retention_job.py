#!/usr/bin/env python3
"""
Retention job (ops-run).
- Hard-deletes old rows to enforce PHI retention.
- Targets: sessions, soap_notes, audit_log.
- Retention window configured via PHI_RETENTION_DAYS (default 90).
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta

import asyncpg

RETENTION_DAYS = int(os.getenv("PHI_RETENTION_DAYS", "90"))
DATABASE_URL = os.getenv("DATABASE_URL")
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retention")


async def delete_or_preview(conn: asyncpg.Connection, table: str, condition: str, cutoff: datetime, id_field: str = "id"):
    """
    Run a delete with RETURNING (or a preview select when DRY_RUN=true).
    condition should contain a single $1 placeholder for cutoff timestamp.
    """
    base = f"{table} WHERE {condition}"
    if DRY_RUN:
        rows = await conn.fetch(f"SELECT {id_field} FROM {base}", cutoff)
    else:
        rows = await conn.fetch(f"DELETE FROM {base} RETURNING {id_field}", cutoff)
    return [r[id_field] for r in rows]


async def main():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL required for retention_job")

    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    logger.info(
        "Starting retention job",
        extra={"retentionDays": RETENTION_DAYS, "cutoff": cutoff.isoformat(), "dryRun": DRY_RUN},
    )

    sessions_condition = """
    (deleted_at IS NOT NULL AND deleted_at < $1)
    OR (archived_at IS NOT NULL AND archived_at < $1)
    OR (deleted_at IS NULL AND archived_at IS NULL AND started_at < $1)
    """
    notes_condition = """
    (deleted_at IS NOT NULL AND deleted_at < $1)
    OR (archived_at IS NOT NULL AND archived_at < $1)
    OR (deleted_at IS NULL AND archived_at IS NULL AND created_at < $1)
    """
    audit_condition = "created_at < $1"

    pool = await asyncpg.create_pool(DATABASE_URL)
    try:
        async with pool.acquire() as conn:
            sess_ids = await delete_or_preview(conn, "sessions", sessions_condition, cutoff)
            note_ids = await delete_or_preview(conn, "soap_notes", notes_condition, cutoff)
            audit_ids = await delete_or_preview(conn, "audit_log", audit_condition, cutoff)
            logger.info(
                "Retention job finished",
                extra={
                    "sessionsRemoved": len(sess_ids),
                    "notesRemoved": len(note_ids),
                    "auditRemoved": len(audit_ids),
                    "dryRun": DRY_RUN,
                },
            )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
