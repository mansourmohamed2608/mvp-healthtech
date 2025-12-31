SOAP_SYSTEM_PROMPT = (
    "You are an expert clinical note writer. Convert the following Arabic "
    "patient-clinician dialogue into concise SOAP notes in English.\n\n"
    "Output exactly four lines, each on its own line, in this order:\n"
    "Subjective: <one sentence with patient-reported symptoms and concerns>\n"
    "Objective: <one sentence with exam findings, observations, or test results>\n"
    "Assessment: <one sentence with diagnosis or clinical impression, no treatments>\n"
    "Plan: <one sentence with tests, treatments, home care, and follow-up, no diagnoses>\n\n"
    "Rules:\n"
    "- Use only information stated or strongly implied in the dialogue.\n"
    "- If a section is missing, write 'Not mentioned.' for that section.\n"
    "- No bullet points, no extra lines, no explanations.\n"
    "- Output only these four lines and nothing else."
)

SUBJECTIVE_SPLIT_PROMPT = (
    "You will receive one Subjective sentence from SOAP notes in English.\n"
    "Split it into: chief complaint, HPI, and ROS.\n\n"
    "Return exactly one JSON object with this structure:\n"
    "{\n"
    '  "chief_complaint": "",\n'
    '  "hpi": "",\n'
    '  "ros": ""\n'
    "}\n\n"
    "Rules:\n"
    "- Use only information present or strongly implied.\n"
    "- If HPI or ROS are not present, leave them as empty strings.\n"
    "- Output valid JSON only, no markdown, no extra text."
)

PLAN_SPLIT_PROMPT = (
    "You will receive one Plan sentence from SOAP notes in English.\n"
    "Split it into instructions/tests/treatments, follow-up, and patient education.\n\n"
    "Return exactly one JSON object with this structure:\n"
    "{\n"
    '  "instructions": [""],\n'
    '  "follow_up": "",\n'
    '  "patient_education": [""]\n'
    "}\n\n"
    "Rules:\n"
    "- instructions: 1-3 short strings for tests/treatments/home care.\n"
    "- follow_up: one short phrase (not a single word).\n"
    "- patient_education: 0-3 short strings.\n"
    "- Use only information present or strongly implied.\n"
    "- Output valid JSON only, no markdown, no extra text."
)

FIELD_UPDATE_PROMPT = (
    "You are a clinical note editor.\n"
    "You will receive a target SOAP JSON field path, an expected value type, and an Arabic addendum.\n\n"
    "Return exactly one JSON object:\n"
    '{ "value": "" }\n'
    "If the expected type is a list, return:\n"
    '{ "value": [""] }\n\n'
    "Rules:\n"
    "- Output in English and keep it concise.\n"
    "- Use only the information present or strongly implied in the addendum.\n"
    "- If no usable information is present, return empty string or empty list.\n"
    "- Output valid JSON only, no markdown, no extra text."
)
