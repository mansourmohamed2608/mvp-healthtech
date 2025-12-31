import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from llm_client import LlmClient
from soap_prompts import (
    SOAP_SYSTEM_PROMPT,
    SUBJECTIVE_SPLIT_PROMPT,
    PLAN_SPLIT_PROMPT,
    FIELD_UPDATE_PROMPT,
)
from template_engine import render_template


@dataclass
class SoapSections:
    subjective: str
    objective: str
    assessment: str
    plan: str


def parse_soap_lines(text: str) -> SoapSections:
    fields = {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        key = label.strip().lower()
        if key in fields:
            fields[key] = value.strip()

    if not any(fields.values()):
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        fields["subjective"] = parts[0] if len(parts) > 0 else ""
        fields["objective"] = parts[1] if len(parts) > 1 else ""
        fields["assessment"] = parts[2] if len(parts) > 2 else ""
        fields["plan"] = parts[3] if len(parts) > 3 else ""

    return SoapSections(
        subjective=fields["subjective"],
        objective=fields["objective"],
        assessment=fields["assessment"],
        plan=fields["plan"],
    )


def extract_json(raw_text: str) -> Tuple[Dict[str, Any] | None, str | None]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "json_not_found"
    snippet = cleaned[start : end + 1]
    try:
        obj = json.loads(snippet)
        if not isinstance(obj, dict):
            return None, "json_not_object"
        return obj, None
    except json.JSONDecodeError:
        return None, "json_decode_error"


def _normalize_list(values: Any, fallback: str = "") -> List[str]:
    if isinstance(values, list):
        items = [v.strip() for v in values if isinstance(v, str) and v.strip()]
        return items
    if isinstance(values, str) and values.strip():
        return [values.strip()]
    return [fallback] if fallback else []


def _merge_string(existing: str, incoming: str, mode: str) -> str:
    existing = (existing or "").strip()
    incoming = (incoming or "").strip()
    if not incoming:
        return existing
    if mode == "replace" or not existing or existing.lower() == "not mentioned.":
        return incoming
    if incoming.lower() in existing.lower():
        return existing
    if existing.endswith((".", "!", "?")):
        return f"{existing} {incoming}"
    return f"{existing}; {incoming}"


def _merge_list(existing: Any, incoming: Any, mode: str) -> List[str]:
    base = _normalize_list(existing)
    if mode == "append":
        add = _normalize_list(incoming)
        return (base + add) or base
    return _normalize_list(incoming)


def _parse_field_path(path: str) -> List[Any]:
    tokens: List[Any] = []
    for segment in path.split("."):
        seg = segment.strip()
        if not seg:
            continue
        match = re.match(r"^([^\[\]]+)(?:\[(\d+)\])?$", seg)
        if not match:
            raise ValueError("invalid_field_path")
        key = match.group(1)
        tokens.append(key)
        if match.group(2) is not None:
            tokens.append(int(match.group(2)))
    return tokens


def _get_nested_value(data: Any, tokens: List[Any]) -> Any:
    cur = data
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(cur, list) or token >= len(cur):
                return None
            cur = cur[token]
        else:
            if not isinstance(cur, dict) or token not in cur:
                return None
            cur = cur[token]
    return cur


def _ensure_container(next_token: Any) -> Any:
    if isinstance(next_token, int):
        return []
    return {}


def _set_nested_value(data: Dict[str, Any], tokens: List[Any], value: Any, mode: str) -> None:
    cur: Any = data
    for idx, token in enumerate(tokens):
        is_last = idx == len(tokens) - 1
        if isinstance(token, int):
            if not isinstance(cur, list):
                raise ValueError("path_expected_list")
            while len(cur) <= token:
                cur.append(_ensure_container(tokens[idx + 1] if not is_last else {}))
            if is_last:
                cur[token] = value
            else:
                if not isinstance(cur[token], (dict, list)):
                    cur[token] = _ensure_container(tokens[idx + 1])
                cur = cur[token]
        else:
            if not isinstance(cur, dict):
                raise ValueError("path_expected_object")
            if is_last:
                existing = cur.get(token)
                if isinstance(existing, list) or isinstance(value, list):
                    cur[token] = _merge_list(existing, value, mode)
                elif isinstance(existing, str) or isinstance(value, str):
                    cur[token] = _merge_string(str(existing or ""), str(value or ""), mode)
                else:
                    cur[token] = value
            else:
                if token not in cur or not isinstance(cur[token], (dict, list)):
                    cur[token] = _ensure_container(tokens[idx + 1])
                cur = cur[token]


def _coerce_value(value: Any, expected: str) -> Any:
    if expected == "list":
        if isinstance(value, list):
            return _normalize_list(value)
        if isinstance(value, str):
            return _normalize_list(value)
        return []
    if isinstance(value, list):
        return "; ".join(_normalize_list(value))
    return str(value) if value is not None else ""


async def generate_field_value(
    llm: LlmClient,
    field_path: str,
    transcript: str,
    expected_type: str,
    existing_value: Any,
    session_id: str | None,
) -> Any:
    if not transcript.strip():
        return "" if expected_type != "list" else []
    existing_snapshot = json.dumps(existing_value, ensure_ascii=False) if existing_value is not None else "null"
    messages = [
        {"role": "system", "content": FIELD_UPDATE_PROMPT},
        {
            "role": "user",
            "content": (
                f"Field path: {field_path}\n"
                f"Expected type: {expected_type}\n"
                f"Existing value: {existing_snapshot}\n\n"
                f"Arabic addendum:\n{transcript}\n\n"
                "Return JSON only."
            ),
        },
    ]
    raw = await llm.generate(messages, max_new_tokens=180, temperature=0.0, session_id=session_id)
    obj, err = extract_json(raw)
    if err or obj is None or "value" not in obj:
        return _coerce_value(transcript, expected_type)
    return _coerce_value(obj.get("value"), expected_type)


def apply_field_update(
    note_json: Dict[str, Any],
    field_path: str,
    value: Any,
    mode: str,
) -> Dict[str, Any]:
    tokens = _parse_field_path(field_path)
    _set_nested_value(note_json, tokens, value, mode)
    return note_json


def parse_field_path(path: str) -> List[Any]:
    return _parse_field_path(path)


def get_field_value(note_json: Dict[str, Any], field_path: str) -> Any:
    tokens = _parse_field_path(field_path)
    return _get_nested_value(note_json, tokens)


def resolve_section_from_path(field_path: str) -> str | None:
    if not field_path:
        return None
    head = field_path.split(".", 1)[0].strip().lower()
    if head in {"subjective", "objective", "assessment", "plan"}:
        return head
    return None


def summarize_value(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(_normalize_list(value))
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value is not None else ""


async def split_subjective(llm: LlmClient, text: str, session_id: str | None) -> Dict[str, str]:
    if not text.strip():
        return {"chief_complaint": "", "hpi": "", "ros": ""}

    messages = [
        {"role": "system", "content": SUBJECTIVE_SPLIT_PROMPT},
        {
            "role": "user",
            "content": f"Subjective sentence:\n{text}\n\nReturn JSON only.",
        },
    ]
    raw = await llm.generate(messages, max_new_tokens=320, temperature=0.0, session_id=session_id)
    obj, err = extract_json(raw)
    if err or obj is None:
        return {"chief_complaint": text, "hpi": "", "ros": ""}
    return {
        "chief_complaint": str(obj.get("chief_complaint", "") or text),
        "hpi": str(obj.get("hpi", "") or ""),
        "ros": str(obj.get("ros", "") or ""),
    }


async def split_plan(llm: LlmClient, text: str, session_id: str | None) -> Dict[str, Any]:
    if not text.strip():
        return {"instructions": [], "follow_up": "", "patient_education": []}

    messages = [
        {"role": "system", "content": PLAN_SPLIT_PROMPT},
        {
            "role": "user",
            "content": f"Plan sentence:\n{text}\n\nReturn JSON only.",
        },
    ]
    raw = await llm.generate(messages, max_new_tokens=320, temperature=0.0, session_id=session_id)
    obj, err = extract_json(raw)
    if err or obj is None:
        return {"instructions": [text], "follow_up": "", "patient_education": []}

    instructions = _normalize_list(obj.get("instructions"), fallback=text)
    education = _normalize_list(obj.get("patient_education"), fallback="")
    follow_up = obj.get("follow_up", "")
    if not isinstance(follow_up, str):
        follow_up = ""
    return {
        "instructions": instructions[:3],
        "follow_up": follow_up.strip(),
        "patient_education": education[:3],
    }


def build_context(
    sections: SoapSections,
    subj_split: Dict[str, str],
    plan_split: Dict[str, Any],
    patient_name: str,
    date_of_visit: str,
    provider_name: str,
) -> Dict[str, Any]:
    return {
        "patient_name": patient_name,
        "date_of_visit": date_of_visit,
        "provider_name": provider_name,
        "subjective": sections.subjective,
        "objective": sections.objective,
        "assessment": sections.assessment,
        "plan": sections.plan,
        "chief_complaint": subj_split.get("chief_complaint", ""),
        "hpi": subj_split.get("hpi", ""),
        "ros": subj_split.get("ros", ""),
        "plan_instructions": plan_split.get("instructions", []),
        "plan_follow_up": plan_split.get("follow_up", ""),
        "plan_education": plan_split.get("patient_education", []),
        "vital_bp": "",
        "vital_hr": "",
        "vital_temp": "",
        "vital_rr": "",
        "vital_spo2": "",
        "objective_cardio": "",
        "objective_resp": "",
        "objective_heent": "",
        "objective_abdomen": "",
        "objective_msk": "",
        "objective_neuro": "",
        "objective_ext": "",
        "icd_codes": [],
        "provider_signature": "",
        "clarification_needed": [
            "Vital signs not documented.",
            "Medications not specified.",
            "ICD-10-AM codes not assigned.",
        ],
    }


async def generate_structured_note(
    llm: LlmClient,
    transcript: str,
    template: Dict[str, Any],
    patient_name: str,
    date_of_visit: str,
    provider_name: str,
    session_id: str | None,
    patient_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context_block = ""
    if patient_context:
        context_block = f"\n\nPatient context:\n{json.dumps(patient_context, ensure_ascii=False)}"
    messages = [
        {"role": "system", "content": SOAP_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Arabic dialogue:\n\n"
                f"{transcript}{context_block}\n\n"
                "Return the SOAP notes in the exact format described."
            ),
        },
    ]
    soap_text = await llm.generate(messages, max_new_tokens=220, temperature=0.0, session_id=session_id)
    sections = parse_soap_lines(soap_text)
    subj_split = await split_subjective(llm, sections.subjective, session_id)
    plan_split = await split_plan(llm, sections.plan, session_id)
    context = build_context(sections, subj_split, plan_split, patient_name, date_of_visit, provider_name)
    note_json = render_template(template, context)
    return {
        "sections": sections,
        "soap_text": soap_text,
        "note_json": note_json,
        "subj_split": subj_split,
        "plan_split": plan_split,
    }
