import warnings
import re

try:
    from transformers import pipeline, set_seed
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

MEDICAL_RANGES = {
    "glucose": {"low": 70, "high": 100, "unit": "mg/dL"},
    "cholesterol": {"low": 0, "high": 200, "unit": "mg/dL"},
    "hemoglobin": {"low": 12, "high": 16, "unit": "g/dL"},
    "blood_pressure_sys": {"low": 90, "high": 120, "unit": "mmHg"},
    "blood_pressure_dia": {"low": 60, "high": 80, "unit": "mmHg"},
    "heart_rate": {"low": 60, "high": 100, "unit": "bpm"},
    "platelet_count": {"low": 150000, "high": 450000, "unit": "mcL"}
}

def grounded_data_builder(features: dict) -> tuple:
    grounded = {}
    trends = {}
    
    for k, v in features.items():
        if k.startswith("current_"):
            param = k.replace("current_", "")
            if param in MEDICAL_RANGES:
                ref = MEDICAL_RANGES[param]
                status = "Normal"
                if v > ref["high"]:
                    status = "High"
                elif v < ref["low"]:
                    status = "Low"
                    
                ref_str = f"{ref['low']}-{ref['high']}" if ref['low'] > 0 else f"<{ref['high']}"
                grounded[param] = f"{v} {ref['unit']} ({status}, Normal: {ref_str} {ref['unit']})"
            else:
                grounded[param] = str(v)
        elif k.startswith("trend_"):
            param = k.replace("trend_", "")
            trends[f"trend_{param}"] = str(v).capitalize()
            
    return grounded, trends

def prompt_builder(risk_level: str, grounded_params: dict, trends: dict) -> str:
    params_str = "\n".join([f"- {k}: {v}" for k, v in grounded_params.items()])
    trends_str = "\n".join([f"- {k.replace('trend_', '')}: {v}" for k, v in trends.items() if v != "Stable"])
    if not trends_str:
        trends_str = "- All parameters stable"
        
    return f"""You are a friendly, empathetic AI medical assistant speaking directly to a patient.

You MUST strictly follow these rules:
* Speak directly to the patient in a warm, comforting tone (e.g., "Hi there! I've looked at your results...").
* Use simple, every-day language. NO complex medical jargon without a simple explanation.
* Only use the provided patient data.
* Do NOT add diseases or conditions not supported by the data.
* Do NOT exaggerate or scare the patient.
* Keep the explanation clear, actionable, and safe.

Patient Data:
Risk Level: {risk_level}

Parameters:
{params_str}

Trends:
{trends_str}

Tasks:
1. Briefly summarize their overall health status in a friendly way based on the risk level. Explain what their risk level means simply.
2. Discuss any numbers (parameters) that are out of place in very simple terms (e.g., instead of "Hyperlipidemia", talk about "higher than normal cholesterol").
3. Give gentle, actionable recommendations.

Output format (do NOT use markdown or special characters):
Clinical Summary:
...
Medical Explanation:
...
Recommendation:
..."""

def output_parser(generated_text: str) -> dict:
    sections = {
        "clinical_summary": "",
        "llm_explanation": "",
        "recommendation": ""
    }
    
    lines = generated_text.split('\n')
    current_key = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Clinical Summary:"):
            current_key = "clinical_summary"
            sections[current_key] += line.replace("Clinical Summary:", "").strip() + " "
        elif line.startswith("Medical Explanation:"):
            current_key = "llm_explanation"
            sections[current_key] += line.replace("Medical Explanation:", "").strip() + " "
        elif line.startswith("Recommendation:"):
            current_key = "recommendation"
            sections[current_key] += line.replace("Recommendation:", "").strip() + " "
        else:
            if current_key:
                sections[current_key] += line + " "
                
    for k in sections:
        sections[k] = sections[k].strip()
        
    return sections

def map_parameters_to_conditions(features: dict) -> list:
    """
    Deterministically maps raw parameter values to valid medical condition names.
    """
    conditions = set()
    
    glucose = features.get("current_glucose")
    if glucose is not None:
        if glucose > 100:
            conditions.add("Hyperglycemia")
        elif glucose < 70:
            conditions.add("Hypoglycemia")
            
    cholesterol = features.get("current_cholesterol")
    if cholesterol is not None:
        if cholesterol > 200:
            conditions.add("Hyperlipidemia")
            
    sys = features.get("current_blood_pressure_sys")
    dia = features.get("current_blood_pressure_dia")
    if sys is not None or dia is not None:
        sys = sys if sys is not None else 120
        dia = dia if dia is not None else 80
        if sys > 120 or dia > 80:
            conditions.add("Hypertension")
        elif sys < 90 or dia < 60:
            conditions.add("Hypotension")
            
    hemo = features.get("current_hemoglobin")
    if hemo is not None:
        if hemo < 12:
            conditions.add("Anemia")
            
    hr = features.get("current_heart_rate")
    if hr is not None:
        if hr > 100:
            conditions.add("Tachycardia")
        elif hr < 60:
            conditions.add("Bradycardia")
            
    return sorted(list(conditions))

def deterministic_fallback(risk_level: str, grounded_params: dict) -> dict:
    abnormal = [k for k, v in grounded_params.items() if "(High" in v or "(Low" in v]
    
    summary = f"Patient assessed as {risk_level} based on current parameter readings."
    
    explanation = f"The {risk_level.lower()} designation reflects abnormal physiological markers. "
    if abnormal:
        param_details = [f"{k} {grounded_params[k].split(' (')[0]}" for k in abnormal]
        explanation += f"Specifically, deviations in {', '.join(param_details)} indicate abnormalities requiring clinical attention."
    else:
        explanation += "Parameters remain within standard reference ranges."
        
    recommendation = "Suggest standard clinical follow-up and monitoring of any noted deviations."
    if risk_level in ["High"]:
        recommendation = "Immediate physician review recommended due to critical parameter deviations."
        
    return {
        "clinical_summary": summary,
        "llm_explanation": explanation,
        "recommendation": recommendation
    }

def safety_validator(parsed_output: dict, risk_level: str, grounded_params: dict, possible_conditions: list) -> dict:
    banned_words = ["cancer", "tumor", "oncology", "malignan", "leukemia", "covid"]
    needs_fallback = False
    
    ALL_KNOWN_CONDITIONS = ["hyperglycemia", "hypoglycemia", "hyperlipidemia", "hypertension", "hypotension", "anemia", "tachycardia", "bradycardia"]
    possible_lower = [c.lower() for c in possible_conditions]
    
    for k, text in parsed_output.items():
        if isinstance(text, str):
            lower_text = text.lower()
            if any(banned in lower_text for banned in banned_words):
                needs_fallback = True
                break
                
            for condition in ALL_KNOWN_CONDITIONS:
                if condition in lower_text and condition not in possible_lower:
                    needs_fallback = True
                    break
                    
            if risk_level == "Low":
                if any(w in lower_text for w in ["severe", "critical", "high risk", "emergency"]):
                    needs_fallback = True
                    break
                    
            if needs_fallback:
                break
                
            # Replace generic/vague terminology with precise clinical terms
            text = text.replace(" imbalance", " abnormalities")
            text = text.replace(" issues", " clinical deviations")
            text = text.replace(" problems", " abnormalities")
            text = text.replace("Imbalance", "Abnormalities")
            text = text.replace("Issues", "Clinical deviations")
            text = text.replace("Problems", "Abnormalities")
            parsed_output[k] = text

    if needs_fallback or len(parsed_output.get("llm_explanation", "")) < 10:
        return deterministic_fallback(risk_level, grounded_params)
        
    return parsed_output

class ReasoningEngine:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"): 
        self.model_name = model_name
        self.generator = None
        
        if TRANSFORMERS_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self.generator = pipeline('text-generation', model=self.model_name, device=-1)
                    set_seed(42)
                except Exception as e:
                    print(f"Warning: Could not load LLM {self.model_name}. Attempting fallback to gpt2.")
                    try:
                        self.generator = pipeline('text-generation', model="gpt2", device=-1)
                    except Exception:
                        pass
                        
    def analyze(self, features: dict, risk_level: str, possible_conditions: list) -> dict:
        grounded_params, trends = grounded_data_builder(features)
        
        # Standardize Risk Label strictly to "Low", "Medium", "High"
        std_risk_level = risk_level.replace(" Risk", "").replace("Severe", "High")
        if std_risk_level not in ["Low", "Medium", "High"]:
            std_risk_level = "Medium"
            
        prompt = prompt_builder(std_risk_level, grounded_params, trends)
        
        raw_output = ""
        if self.generator:
            try:
                outputs = self.generator(
                    prompt, 
                    max_new_tokens=150, 
                    temperature=0.3, 
                    top_p=0.8,
                    do_sample=True,
                    repetition_penalty=1.2,
                    return_full_text=False
                )
                raw_output = outputs[0]['generated_text'].strip()
            except Exception:
                pass
                
        parsed = output_parser(raw_output)
        safe_output = safety_validator(parsed, std_risk_level, grounded_params, possible_conditions)
        
        # Build trend_analysis string
        trends_str = ", ".join([f"{k.replace('trend_', '')} is {v.lower()}" for k, v in trends.items() if v != "Stable"])
        if not trends_str:
            trends_str = "All parameters are stable."
        else:
            trends_str = trends_str.capitalize() + "."
            
        return {
            "trend_analysis": trends_str,
            "clinical_summary": safe_output["clinical_summary"],
            "llm_explanation": safe_output["llm_explanation"],
            "recommendation": safe_output["recommendation"]
        }
