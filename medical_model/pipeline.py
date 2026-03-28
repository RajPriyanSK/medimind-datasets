from .feature_engineering import compute_features
from .model import RiskModel
from .llm_inference import ReasoningEngine, condition_mapper, explicit_hidden_risk_detector, calculate_severity, contraindication_detector

_risk_model = RiskModel()
_reasoning_engine = ReasoningEngine()

def evaluate_data_quality(patient_data: dict, features: dict) -> str:
    history = patient_data.get("history", [])
    report_type = patient_data.get("report_type", "unknown")
    
    score = 0
    
    if len(history) > 0:
        score += 1
        
    if len(history) >= 2:
        score += 1
        
    if report_type == "lab":
        if "current_glucose" in features and "current_cholesterol" in features and "current_blood_pressure_sys" in features:
            score += 1
    elif report_type == "prescription":
        if "medication" in patient_data.get("current_report", {}):
            score += 1
    elif report_type == "scan":
        if "findings" in patient_data.get("current_report", {}):
            score += 1
            
    if score >= 2:
        return "high"
    elif score == 1:
        return "medium"
    else:
        return "low"

def generate_confidence_factors(features: dict, history: list, possible_conditions: list) -> dict:
    from .llm_inference import MEDICAL_RANGES
    severe_count = 0
    moderate_count = 0
    mild_count = 0
    
    for k, v in features.items():
        if k.startswith("current_") and isinstance(v, (int, float)):
            param = k.replace("current_", "")
            if param in MEDICAL_RANGES:
                ref = MEDICAL_RANGES[param]
                sev = calculate_severity(v, ref)
                if sev == "critical": severe_count += 1
                elif sev == "moderate": moderate_count += 1
                elif sev == "mild": mild_count += 1
                
    return {
        "critical_abnormalities": severe_count,
        "moderate_abnormalities": moderate_count,
        "mild_abnormalities": mild_count,
        "history_length": len(history),
        "primary_conditions_detected": len(possible_conditions)
    }

def generate_confidence_reason(factors: dict, trends: str, possible_conditions: list) -> str:
    abnormal_count = factors["critical_abnormalities"] + factors["moderate_abnormalities"] + factors["mild_abnormalities"]
    hist_len = factors["history_length"]
    severe_count = factors["critical_abnormalities"]
    
    if abnormal_count == 0:
        base = "High confidence as all evaluated parameters remain strictly within biological reference ranges."
    else:
        severity_str = "critical " if severe_count > 0 else ""
        mapped_conds = ', '.join(possible_conditions) if possible_conditions else "non-specific findings"
        base = f"High confidence due to {abnormal_count} {severity_str}abnormalities mapping to {mapped_conds}."
        
    if hist_len > 0:
        trend_dir = "worsening" if "increasing" in trends.lower() else "stable"
        base += f" Reasoning is strongly supported by {trend_dir} trends across {hist_len} historical sequence(s)."
    else:
        base += " Reasoning relies exclusively on baseline cross-sectional data due to absent history."
        
    return base

def predict(patient_data: dict) -> dict:
    try:
        current_report = patient_data.get("current_report", {})
        history = patient_data.get("history", [])
        report_type = patient_data.get("report_type", "unknown")
        
        # 1. Feature Engineering
        features = compute_features(patient_data)
        
        # 2. Risk Prediction Model
        raw_risk_level, confidence = _risk_model.predict(features)
        
        # Standardize Risk Label strictly to "Low", "Medium", "High"
        risk_level = raw_risk_level.replace(" Risk", "").replace("Severe", "High")
        if risk_level not in ["Low", "Medium", "High"]:
            risk_level = "Medium"
            
        # 3. Extract Possible Conditions Cleanly
        possible_conditions = condition_mapper(features, current_report, report_type)
        primary_condition = possible_conditions[0] if possible_conditions else "Unknown primary condition"
        
        # 4. Explicit Hidden Risk & Contraindication Detectors
        explicit_hidden_risks = explicit_hidden_risk_detector(features, possible_conditions)
        contraindications = contraindication_detector(features, current_report)
        
        # 5. Data Quality Scoring
        data_quality = evaluate_data_quality(patient_data, features)
        if data_quality == "low":
            confidence = max(0.0, confidence - 0.2)
            
        # 6. Grounded LLM Reasoning Engine
        reasoning = _reasoning_engine.analyze(
            patient_data, 
            risk_level, 
            possible_conditions, 
            explicit_hidden_risks, 
            contraindications
        )
        
        # 7. Generate Confidence Factors & Reason
        confidence_factors = generate_confidence_factors(features, history, possible_conditions)
        confidence_reason = generate_confidence_reason(confidence_factors, reasoning.get("trend_analysis", ""), possible_conditions)
        
        if data_quality == "low":
            confidence_reason += " (Note: Confidence metric was reduced due to LOW data quality and tracking gaps.)"
        
        # 8. Output Formatting (Ensuring every required key is strictly present)
        response = {
            "risk_level": risk_level,
            "confidence": round(confidence, 2),
            "confidence_reason": confidence_reason,
            "confidence_factors": confidence_factors,
            "data_quality": data_quality,
            "trend_analysis": reasoning.get("trend_analysis", ""),
            "detected_abnormalities": reasoning.get("detected_abnormalities", []),
            "hidden_risks": reasoning.get("hidden_risks", []),
            "contraindications": contraindications,
            "future_risk_prediction": reasoning.get("future_risk_prediction", ""),
            "health_timeline": reasoning.get("health_timeline", ""),
            "primary_condition": primary_condition,
            "possible_conditions": possible_conditions,
            "doctor_summary": reasoning.get("doctor_summary", ""),
            "patient_friendly_explanation": reasoning.get("patient_friendly_explanation", ""),
            "recommendation": reasoning.get("recommendation", "")
        }
        
        return response
    except Exception as e:
        return {
            "risk_level": "Unknown",
            "confidence": 0.0,
            "confidence_reason": f"System error occurred: {str(e)}",
            "confidence_factors": {},
            "data_quality": "low",
            "trend_analysis": "Error computing trends",
            "detected_abnormalities": [],
            "hidden_risks": [],
            "contraindications": [],
            "future_risk_prediction": "Error",
            "health_timeline": "Error",
            "primary_condition": "Unknown",
            "possible_conditions": [],
            "doctor_summary": f"System error occurred: {str(e)}",
            "patient_friendly_explanation": "An unexpected error disrupted the analysis. Please consult a doctor.",
            "recommendation": "System error occurred."
        }
