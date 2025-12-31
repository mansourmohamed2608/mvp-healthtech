import re
from typing import Any, Dict

PLACEHOLDER_RE = re.compile(r"\{\{([a-zA-Z0-9_]+)\}\}")


def has_placeholders(value: Any) -> bool:
    if isinstance(value, str):
        return bool(PLACEHOLDER_RE.search(value))
    if isinstance(value, list):
        return any(has_placeholders(v) for v in value)
    if isinstance(value, dict):
        return any(has_placeholders(v) for v in value.values())
    return False


def render_template(template: Any, context: Dict[str, Any]) -> Any:
    if isinstance(template, dict):
        return {k: render_template(v, context) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(v, context) for v in template]
    if isinstance(template, str):
        exact = PLACEHOLDER_RE.fullmatch(template.strip())
        if exact:
            key = exact.group(1)
            return context.get(key, "")

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            val = context.get(key, "")
            if isinstance(val, (list, dict)):
                return str(val)
            return str(val)

        return PLACEHOLDER_RE.sub(replace, template)
    return template
