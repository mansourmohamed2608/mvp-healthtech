SOAP_SYSTEM_PROMPT = (
    "You are an expert clinical note writer. Convert the Arabic patient-clinician dialogue into English SOAP notes.\n\n"
    "Here is an example of correct output:\n"
    "Subjective: Patient is a 52-year-old male presenting with 3 days of sharp chest pain radiating to the left arm, worsening with exertion.\n"
    "Objective: Blood pressure 145/92, heart rate 88 bpm, no murmurs on cardiac auscultation, mild diaphoresis noted.\n"
    "Assessment: Probable stable angina; acute coronary syndrome must be ruled out.\n"
    "Plan: Order 12-lead ECG and troponin levels, prescribe aspirin 325 mg stat, follow up within one week.\n\n"
    "Now write SOAP notes for the provided dialogue using the same format.\n"
    "Rules:\n"
    "- Write exactly four labeled lines: Subjective, Objective, Assessment, Plan.\n"
    "- Each line must begin with the label followed by a colon and a space, then the actual clinical content.\n"
    "- Use only information stated or strongly implied in the dialogue.\n"
    "- If a section has no information in the dialogue, write 'Not mentioned.' after the label.\n"
    "- No bullet points, no extra lines, no preamble, no explanations.\n"
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
