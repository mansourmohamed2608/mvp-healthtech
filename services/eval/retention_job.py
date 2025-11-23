#!/usr/bin/env python3
"""
Retention job (manual/ops-run):
- Marks old sessions and soap_notes as archived (soft delete) based on PHI_RETENTION_DAYS.
- Intended to be run via cron/k8s CronJob by ops; not scheduled inside services.
"""
import os
import asyncio
import asyncpg
from datetime import datetime, timedelta

RETENTION_DAYS = int(os.getenv("PHI_RETENTION_DAYS", "365"))
DATABASE_URL = os.getenv("DATABASE_URL")


async def archive_old(conn, table: str, date_field: str, id_field: str = "id"):
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    rows = await conn.fetch(
        f"UPDATE {table} SET archived_at = now() WHERE {date_field} < $1 AND archived_at IS NULL RETURNING {id_field}",
        cutoff,
    )
    return [r[id_field] for r in rows]


async def main():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL required for retention_job")
    pool = await asyncpg.create_pool(DATABASE_URL)
    async with pool.acquire() as conn:
        sess_ids = await archive_old(conn, "sessions", "started_at", "id")
        note_ids = await archive_old(conn, "soap_notes", "created_at", "id")
        print(f"Archived sessions: {len(sess_ids)} ids={sess_ids}")
        print(f"Archived soap_notes: {len(note_ids)} ids={note_ids}")
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
