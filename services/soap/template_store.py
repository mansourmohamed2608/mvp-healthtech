import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

import asyncpg


class TemplateStore:
    def __init__(self, pool: asyncpg.Pool | None):
        self.pool = pool

    async def ensure_system_templates(self, templates: List[Dict[str, Any]]) -> None:
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            for template in templates:
                await conn.execute(
                    """
                    INSERT INTO soap_templates (id, name, template, is_system)
                    VALUES ($1, $2, $3, true)
                    ON CONFLICT (id) DO UPDATE SET
                      name = EXCLUDED.name,
                      template = EXCLUDED.template,
                      updated_at = now()
                    WHERE soap_templates.is_system = true
                    """,
                    template["id"],
                    template["name"],
                    json.dumps(template["template"]),
                )

    async def list_templates(self) -> List[Dict[str, Any]]:
        if not self.pool:
            return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, is_system, created_at FROM soap_templates ORDER BY created_at DESC"
            )
        return [dict(r) for r in rows]

    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        if not self.pool:
            return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, name, template, is_system FROM soap_templates WHERE id = $1",
                template_id,
            )
        if not row:
            return None
        template = row["template"]
        return {
            "id": row["id"],
            "name": row["name"],
            "template": template if isinstance(template, dict) else json.loads(template),
            "is_system": row["is_system"],
        }

    async def create_template(
        self,
        name: str,
        template: Dict[str, Any],
        created_by: str | None = None,
        template_id: str | None = None,
    ) -> str:
        if not self.pool:
            raise RuntimeError("Template store unavailable")
        template_id = template_id or f"tpl_{uuid4().hex}"
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO soap_templates (id, name, template, created_by, is_system)
                VALUES ($1, $2, $3, $4, false)
                ON CONFLICT (id) DO UPDATE SET
                  name = EXCLUDED.name,
                  template = EXCLUDED.template,
                  created_by = EXCLUDED.created_by,
                  updated_at = now()
                """,
                template_id,
                name,
                json.dumps(template),
                created_by,
            )
        return template_id
