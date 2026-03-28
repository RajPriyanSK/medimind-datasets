import warnings
import re

try:
    from transformers import pipeline, set_seed
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

MEDICAL_RANGES = {
    "glucose": {"low": 70, "high": 126, "unit": "mg/dL"},
    "cholesterol": {"low": 0, "high": 200, "unit": "mg/dL"},
    "hemoglobin": {"low": 12, "high": 16, "unit": "g/dL"},
    "blood_pressure_sys": {"low": 90, "high": 120, "unit": "mmHg"},
    "blood_pressure_dia": {"low": 60, "high": 80, "unit": "mmHg"},
    "heart_rate": {"low": 60, "high": 100, "unit": "bpm"},
    "platelet_count": {"low": 150000, "high": 450000, "unit": "mcL"}
}

DRUG_DB = {
    "metformin": {"class": "Antidiabetic", "condition": "Diabetes", "type": "chronic"},
    "insulin": {"class": "Antidiabetic", "condition": "Diabetes", "type": "chronic"},
    "lisinopril": {"class": "ACE Inhibitor", "condition": "Hypertension", "type": "chronic"},
    "losartan": {"class": "ARB", "condition": "Hypertension", "type": "chronic"},
    "amlodipine": {"class": "Calcium Channel Blocker", "condition": "Hypertension", "type": "chronic"},
    "statin": {"class": "Lipid-lowering agent", "condition": "Hyperlipidemia", "type": "chronic"},
    "atorvastatin": {"class": "Lipid-lowering agent", "condition": "Hyperlipidemia", "type": "chronic"},
    "iron": {"class": "Supplement", "condition": "Anemia", "type": "acute"}
}

def calculate_severity(val: float, ref: dict) -> str:
    if val > ref["high"]:
        ratio = val / ref["high"]
    elif val < ref["low"]:
        ratio = ref["low"] / val if val > 0 else 2.0
    else:
        return "normal"
        
    if ratio > 1.5:
        return "critical"
    elif ratio > 1.2:
        return "moderate"
    else:
        return "mild"

def explicit_hidden_risk_detector(features: dict, primary_conditions: list) -> list:
    hidden_risks = []
    
    condition_to_params = {
        "Hyperglycemia": ["glucose"],
        "Hypoglycemia": ["glucose"],
        "Diabetes": ["glucose"],
        "Hyperlipidemia": ["cholesterol"],
        "Hypertension": ["blood_pressure_sys", "blood_pressure_dia"],
        "Hypotension": ["blood_pressure_sys", "blood_pressure_dia"],
        "Anemia": ["hemoglobin"],
        "Tachycardia": ["heart_rate"],
        "Bradycardia": ["heart_rate"]
    }
    
    primary_params = set()
    for cond in primary_conditions:
        for p in condition_to_params.get(cond, []):
            primary_params.add(p)
            
    for param, ref in MEDICAL_RANGES.items():
        if param in primary_params:
            continue
        val = features.get(f"current_{param}")
        if val is not None and isinstance(val, (int, float)):
            sev = calculate_severity(val, ref)
            if sev != "normal":
                hidden_risks.append(f"{param.replace('_', ' ').title()} is abnormal ({sev.upper()}): {val} {ref['unit']}")
                
    return hidden_risks

def contraindication_detector(features: dict, current_report: dict) -> list:
    contraindications = []
    meds = str(current_report.get("medication", "")).lower()
    
    glucose = features.get("current_glucose")
    sys_bp = features.get("current_blood_pressure_sys")
    dia_bp = features.get("current_blood_pressure_dia")
    
    if glucose is not None and glucose < 70:
        if "metformin" in meds or "insulin" in meds:
            contraindications.append("Active Antidiabetic medication detected alongside acute Hypoglycemia.")
            
    if (sys_bp is not None and sys_bp < 90) or (dia_bp is not None and dia_bp < 60):
        if "lisinopril" in meds or "losartan" in meds or "amlodipine" in meds:
            contraindications.append("Active Antihypertensive medication detected alongside acute Hypotension.")
            
    if "lisinopril" in meds and "losartan" in meds:
        contraindications.append("Contraindication: Concurrent use of ACE Inhibitor (lisinopril) and ARB (losartan) detected.")
        
    return contraindications

def condition_mapper(features: dict, current_report: dict = None, report_type: str = "lab") -> list:
    conditions = set()
    current_report = current_report or {}
    
    if report_type == "lab":
        glucose = features.get("current_glucose")
        if glucose is not None:
            if glucose > 126: conditions.add("Hyperglycemia")
            elif glucose < 70: conditions.add("Hypoglycemia")
                
        cholesterol = features.get("current_cholesterol")
        if cholesterol is not None:
            if cholesterol > 200: conditions.add("Hyperlipidemia")
                
        sys = features.get("current_blood_pressure_sys")
        dia = features.get("current_blood_pressure_dia")
        if sys is not None or dia is not None:
            sys = sys if sys is not None else 120
            dia = dia if dia is not None else 80
            if sys > 120 or dia > 80: conditions.add("Hypertension")
            elif sys < 90 or dia < 60: conditions.add("Hypotension")
                
        hemo = features.get("current_hemoglobin")
        if hemo is not None:
            if hemo < 12: conditions.add("Anemia")
                
        hr = features.get("current_heart_rate")
        if hr is not None:
            if hr > 100: conditions.add("Tachycardia")
            elif hr < 60: conditions.add("Bradycardia")

    elif report_type == "prescription":
        meds = str(current_report.get("medication", "")).lower()
        for drug, info in DRUG_DB.items():
            if drug in meds:
                conditions.add(info["condition"])
        
    elif report_type == "scan":
        findings = str(current_report.get("findings", "")).lower() + " " + str(current_report.get("impression", "")).lower()
        if "tumor" in findings or "mass" in findings: conditions.add("Possible Mass/Tumor")
        if "fracture" in findings: conditions.add("Fracture")
        if "inflammation" in findings: conditions.add("Inflammation")
        if "ischemia" in findings: conditions.add("Ischemia")

    return sorted(list({c.title() for c in conditions}))

def grounded_data_builder(current_report: dict) -> dict:
    grounded = {}
    for param, v in current_report.items():
        if param in MEDICAL_RANGES and isinstance(v, (int, float)) and not isinstance(v, bool):
            ref = MEDICAL_RANGES[param]
            sev = calculate_severity(v, ref)
            status = f"Abnormal - {sev.title()}" if sev != "normal" else "Normal"
                
            ref_str = f"{ref['low']}-{ref['high']}" if ref['low'] > 0 else f"<{ref['high']}"
            grounded[param] = f"{v} {ref['unit']} ({status}, Normal: {ref_str} {ref['unit']})"
        else:
            grounded[param] = str(v)
    return grounded

def context_builder(patient_data: dict, explicit_hidden_risks: list, contraindications: list) -> str:
    from .feature_engineering import compute_features
    features = compute_features(patient_data)
    
    current_report = patient_data.get("current_report", {})
    history = patient_data.get("history", [])
    report_type = patient_data.get("report_type", "unknown")
    personalization = patient_data.get("personalization", {})
    
    grounded_current = grounded_data_builder(current_report)
    
    past_diseases = set()
    for record in history:
        for k in ["diagnoses", "conditions", "disease"]:
            if k in record:
                val = record[k]
                if isinstance(val, list): past_diseases.update([v.title() for v in val])
                elif isinstance(val, str): past_diseases.add(val.title())
    
    context_lines = []
    
    if personalization:
        context_lines.append("Patient Personalization:")
        for k, v in personalization.items():
            context_lines.append(f"- {k.title()}: {v}")
        context_lines.append("")
        
    if report_type == "lab":
        context_lines.append("Report Type: LAB (Focus: Evaluate numerical values, biomarkers, and clinical ranges strictly)")
        context_lines.append("\nCurrent Parameters:")
        for k, v in grounded_current.items():
            trend = features.get(f"trend_{k}", "stable")
            avg = features.get(f"weighted_avg_{k}", "N/A")
            if trend != "stable" and avg != "N/A":
                 context_lines.append(f"- {k.replace('_', ' ').title()}: {v} (Trend: {trend.capitalize()}, Weighted Hist. Avg: {avg})")
            else:
                 context_lines.append(f"- {k.replace('_', ' ').title()}: {v}")
    elif report_type == "prescription":
        context_lines.append("Report Type: PRESCRIPTION (Focus: Extract specific medication strings and carefully infer treated clinical conditions)")
        context_lines.append("\nPrescription Details:")
        meds = str(current_report.get("medication", "")).lower()
        for k, v in current_report.items():
            context_lines.append(f"- {str(k).replace('_', ' ').title()}: {v}")
        
        extracted_drugs = []
        for drug, info in DRUG_DB.items():
            if drug in meds:
                extracted_drugs.append(f"{drug.title()} (Class: {info['class']}, Treats: {info['condition']}, Type: {info['type']})")
        if extracted_drugs:
            context_lines.append("\nIdentified Treatments:")
            for d in extracted_drugs:
                context_lines.append(f"- {d}")
                
    elif report_type == "scan":
        context_lines.append("Report Type: SCAN (Focus: Interpret descriptive text-based scan interpretation objectively. This is a text-based scan interpretation, not direct image analysis. Do not imply visual understanding.)")
        context_lines.append("\nScan Findings:")
        for k, v in current_report.items():
            context_lines.append(f"- {str(k).replace('_', ' ').title()}: {v}")
    else:
        context_lines.append(f"Report Type: {str(report_type).upper()}")
        for k, v in current_report.items():
            context_lines.append(f"- {k}: {v}")
            
    if explicit_hidden_risks:
        context_lines.append("\nExplicitly Detected Hidden Risks (Not part of primary condition):")
        for hr in explicit_hidden_risks:
            context_lines.append(f"- {hr}")
            
    if contraindications:
        context_lines.append("\nCRITICAL - Pharmacological Contraindications Detected:")
        for ci in contraindications:
            context_lines.append(f"- {ci}")
             
    if past_diseases:
        context_lines.append("\nPast Disease Records (Including Cured):")
        for d in sorted(list(past_diseases)):
            context_lines.append(f"- {d}")
            
    return "\n".join(context_lines)

def prompt_builder(risk_level: str, context: str) -> str:
    return f"""You are a clinical medical assistant evaluating a patient.

You MUST follow ALL rules:
* Only use provided patient data
* Do NOT invent diseases
* Do NOT exaggerate conditions
* Do NOT include unrelated risks
* Use numerical values explicitly
* Base reasoning heavily on provided trends and chronological history

You MUST:
* Identify abnormal parameters, and attach explicit severity tags (mild/moderate/critical) to each in Detected Abnormalities
* Generate a future risk prediction highly grounded in historical progression (e.g. 'Rising glucose trend may indicate...')
* Generate a health timeline summarizing cross-visit progression (must start like: 'Over the chronological history...')
* Mention identified contraindications severely within Recommendation and Doctor Summary

Patient Data:
Risk Level: {risk_level}

Context:
{context}

Return output EXACTLY in this format:

Detected Abnormalities:
...

Hidden Risks:
...

Future Risk Prediction:
...

Health Timeline:
...

Doctor Summary:
...

Patient Friendly Explanation:
...

Recommendation:
..."""

def clean_list(lst: list) -> list:
    seen = set()
    cleaned = []
    for item in lst:
        i_clean = item.strip()
        if i_clean and i_clean.lower() not in ["none", "n/a", "no", "nothing"]:
            if i_clean.lower() not in seen:
                seen.add(i_clean.lower())
                cleaned.append(i_clean)
    return cleaned

def output_parser(generated_text: str) -> dict:
    sections = {
        "detected_abnormalities": [],
        "hidden_risks": [],
        "future_risk_prediction": "",
        "health_timeline": "",
        "doctor_summary": "",
        "patient_friendly_explanation": "",
        "recommendation": ""
    }
    
    current_key = None
    lines = generated_text.split('\n')
    
    for line in lines:
        line_strip = line.strip()
        if not line_strip:
            continue
            
        if line_strip.startswith("Detected Abnormalities:"):
            current_key = "detected_abnormalities"
            content = line_strip.replace("Detected Abnormalities:", "").strip()
            if content: sections[current_key].append(content)
        elif line_strip.startswith("Hidden Risks:"):
            current_key = "hidden_risks"
            content = line_strip.replace("Hidden Risks:", "").strip()
            if content: sections[current_key].append(content)
        elif line_strip.startswith("Future Risk Prediction:"):
            current_key = "future_risk_prediction"
            content = line_strip.replace("Future Risk Prediction:", "").strip()
            if content: sections[current_key] += content + " "
        elif line_strip.startswith("Health Timeline:"):
            current_key = "health_timeline"
            content = line_strip.replace("Health Timeline:", "").strip()
            if content: sections[current_key] += content + " "
        elif line_strip.startswith("Doctor Summary:"):
            current_key = "doctor_summary"
            content = line_strip.replace("Doctor Summary:", "").strip()
            if content: sections[current_key] += content + " "
        elif line_strip.startswith("Patient Friendly Explanation:"):
            current_key = "patient_friendly_explanation"
            content = line_strip.replace("Patient Friendly Explanation:", "").strip()
            if content: sections[current_key] += content + " "
        elif line_strip.startswith("Recommendation:"):
            current_key = "recommendation"
            content = line_strip.replace("Recommendation:", "").strip()
            if content: sections[current_key] += content + " "
        else:
            if current_key in ["detected_abnormalities", "hidden_risks"]:
                if line_strip.startswith("- ") or line_strip.startswith("* "):
                    sections[current_key].append(line_strip[2:].strip())
                else:
                    sections[current_key].append(line_strip)
            elif current_key in ["future_risk_prediction", "health_timeline", "doctor_summary", "patient_friendly_explanation", "recommendation"]:
                sections[current_key] += line_strip + " "
                
    for k in ["future_risk_prediction", "health_timeline", "doctor_summary", "patient_friendly_explanation", "recommendation"]:
        if isinstance(sections[k], str):
             sections[k] = sections[k].strip()
        else:
             sections[k] = " ".join(sections[k]).strip()
        
    for k in ["detected_abnormalities", "hidden_risks"]:
        sections[k] = clean_list(sections[k])
        
    return sections

def safety_validator(parsed_output: dict, possible_conditions: list, history: list) -> dict:
    required_keys = ["detected_abnormalities", "hidden_risks", "future_risk_prediction", "health_timeline", "doctor_summary", "patient_friendly_explanation", "recommendation"]
    for k in required_keys:
        if not parsed_output.get(k):
            parsed_output[k] = [] if k in ["detected_abnormalities", "hidden_risks"] else "Pending specific data analysis."
            
    allowed_conditions = set(c.lower() for c in possible_conditions)
    for record in history:
        for k in ["diagnoses", "conditions", "disease"]:
            if k in record:
                val = record[k]
                if isinstance(val, list): allowed_conditions.update([v.lower() for v in val])
                elif isinstance(val, str): allowed_conditions.add(val.lower())
                
    MAJOR_CONDITIONS = ["hyperglycemia", "hypoglycemia", "hyperlipidemia", "hypertension", "hypotension", "anemia", "tachycardia", "bradycardia", "diabetes", "tumor", "mass", "fracture", "inflammation", "ischemia", "cancer", "leukemia", "covid"]

    for k in ["detected_abnormalities", "hidden_risks"]:
        filtered_list = []
        for item in parsed_output[k]:
            item_lower = item.lower()
            unsupported = False
            for cond in MAJOR_CONDITIONS:
                if cond in item_lower and cond not in allowed_conditions:
                    unsupported = True
                    break
            if not unsupported:
                filtered_list.append(item)
        parsed_output[k] = filtered_list
        
    for k in ["doctor_summary", "patient_friendly_explanation"]:
        text = parsed_output[k]
        text_lower = text.lower()
        for cond in MAJOR_CONDITIONS:
            if cond in text_lower and cond not in allowed_conditions:
                pattern = re.compile(rf"\b{re.escape(cond)}\b", re.IGNORECASE)
                text = pattern.sub("a clinically relevant condition", text)
        parsed_output[k] = text
        
    return parsed_output

class ReasoningEngine:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"): 
        self.model_name = model_name
        self.generator = None
        
        if TRANSFORMERS_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.generator = pipeline('text-generation', model=self.model_name, device=-1, trust_remote_code=True)
                    set_seed(42)
                except Exception as e:
                    print(f"Warning: Could not load LLM {self.model_name}. Error: {e}")
                    print("Attempting fallback to gpt2.")
                    try:
                        self.generator = pipeline('text-generation', model="gpt2", device=-1)
                    except Exception:
                        pass
                        
    def analyze(self, patient_data: dict, risk_level: str, possible_conditions: list, explicit_hidden_risks: list, contraindications: list) -> dict:
        context = context_builder(patient_data, explicit_hidden_risks, contraindications)
        
        std_risk_level = risk_level.replace(" Risk", "").replace("Severe", "High")
        if std_risk_level not in ["Low", "Medium", "High"]:
            std_risk_level = "Medium"
            
        prompt = prompt_builder(std_risk_level, context)
        
        raw_output = ""
        if self.generator:
            try:
                outputs = self.generator(
                    prompt, 
                    max_new_tokens=400, 
                    temperature=0.2, 
                    top_p=0.8,
                    do_sample=True,
                    repetition_penalty=1.2,
                    return_full_text=False
                )
                raw_output = outputs[0]['generated_text'].strip()
            except Exception:
                pass
                
        parsed = output_parser(raw_output)
        safe_output = safety_validator(parsed, possible_conditions, patient_data.get("history", []))
        
        from .feature_engineering import compute_features
        features = compute_features(patient_data)
        trends_lst = [f"{k.replace('trend_', '').replace('_', ' ').title()} is {v}" for k, v in features.items() if k.startswith("trend_") and v != "stable"]
        trends_str = ", ".join(trends_lst)
        if not trends_str:
            trends_str = "All parameters maintain stable progression trajectories."
        else:
            trends_str = trends_str.capitalize() + "."
            
        combined_hidden_risks = clean_list(explicit_hidden_risks + safe_output.get("hidden_risks", []))
        
        abnormalities_joined = " ".join([a.lower() for a in safe_output.get("detected_abnormalities", [])])
        final_hidden_risks = []
        for hr in combined_hidden_risks:
            hr_core = hr.split(" is abnormal")[0].lower().strip()
            if hr_core not in abnormalities_joined:
                final_hidden_risks.append(hr)
            
        return {
            "trend_analysis": trends_str,
            "detected_abnormalities": safe_output.get("detected_abnormalities", []),
            "hidden_risks": final_hidden_risks,
            "future_risk_prediction": safe_output.get("future_risk_prediction", ""),
            "health_timeline": safe_output.get("health_timeline", ""),
            "doctor_summary": safe_output.get("doctor_summary", ""),
            "patient_friendly_explanation": safe_output.get("patient_friendly_explanation", ""),
            "recommendation": safe_output.get("recommendation", "")
        }
