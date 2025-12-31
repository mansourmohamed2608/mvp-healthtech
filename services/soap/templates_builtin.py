DEFAULT_TEMPLATE = {
    "Patient Name": "{{patient_name}}",
    "Date of Visit": "{{date_of_visit}}",
    "Provider Name and Credentials": "{{provider_name}}",
    "Subjective": {
        "Chief Complaint": "{{chief_complaint}}",
        "HPI": "{{hpi}}",
        "ROS": "{{ros}}",
        "Raw": "{{subjective}}",
    },
    "Objective": {
        "Clinical Examination Findings": {
            "Vital Signs": {
                "BP": "{{vital_bp}}",
                "HR": "{{vital_hr}}",
                "Temp": "{{vital_temp}}",
                "RR": "{{vital_rr}}",
                "SpO2": "{{vital_spo2}}",
            },
            "Physical Exam": {
                "General": "{{objective}}",
                "Cardiovascular": "{{objective_cardio}}",
                "Respiratory": "{{objective_resp}}",
                "HEENT": "{{objective_heent}}",
                "Abdomen": "{{objective_abdomen}}",
                "Musculoskeletal": "{{objective_msk}}",
                "Neurological": "{{objective_neuro}}",
                "Extremities": "{{objective_ext}}",
            },
        },
        "Observations": "{{objective}}",
    },
    "Assessment": [{"Diagnosis": "{{assessment}}"}],
    "Plan": {
        "Instructions": "{{plan_instructions}}",
        "Follow-Up": "{{plan_follow_up}}",
        "Patient Education": "{{plan_education}}",
    },
    "ICD-10-AM Codes": "{{icd_codes}}",
    "Provider Signature": "{{provider_signature}}",
    "Clarification Needed": "{{clarification_needed}}",
}

COMPACT_TEMPLATE = {
    "Subjective": "{{subjective}}",
    "Objective": "{{objective}}",
    "Assessment": "{{assessment}}",
    "Plan": {
        "Instructions": "{{plan_instructions}}",
        "Follow-Up": "{{plan_follow_up}}",
        "Patient Education": "{{plan_education}}",
    },
}

SYSTEM_TEMPLATES = [
    {
        "id": "pdf_style_v1",
        "name": "PDF Style",
        "template": DEFAULT_TEMPLATE,
    },
    {
        "id": "compact_v1",
        "name": "Compact SOAP",
        "template": COMPACT_TEMPLATE,
    },
]
