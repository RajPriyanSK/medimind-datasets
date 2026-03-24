from .feature_engineering import compute_features
from .model import RiskModel
from .llm_inference import ReasoningEngine, map_parameters_to_conditions

_risk_model = RiskModel()
_reasoning_engine = ReasoningEngine()

def generate_confidence_reason(features: dict, trends: str) -> str:
    """
    Generates deterministic confidence reasoning based on features.
    """
    abnormal_count = 0
    if features.get('current_glucose', 0) > 100 or features.get('current_glucose', 100) < 70:
        abnormal_count += 1
    if features.get('current_cholesterol', 0) > 200:
        abnormal_count += 1
    if features.get('current_blood_pressure_sys', 120) > 120 or features.get('current_blood_pressure_dia', 80) > 80:
        abnormal_count += 1
    if features.get('current_hemoglobin', 14) < 12:
        abnormal_count += 1
        
    if abnormal_count == 0:
        return "No parameters are significantly outside normal ranges."
    elif abnormal_count == 1:
        base = "One parameter is significantly outside normal ranges"
    else:
        base = "Multiple parameters are significantly outside normal ranges"
        
    if "increasing" in trends.lower() or "decreasing" in trends.lower():
        return base + " with corresponding worsening trends."
    return base + "."

def predict(patient_data: dict) -> dict:
    try:
        # 1. Feature Engineering
        features = compute_features(patient_data)
        
        # 2. Risk Prediction Model
        raw_risk_level, confidence = _risk_model.predict(features)
        
        # Standardize Risk Label strictly to "Low", "Medium", "High"
        risk_level = raw_risk_level.replace(" Risk", "").replace("Severe", "High")
        if risk_level not in ["Low", "Medium", "High"]:
            risk_level = "Medium"
            
        # 3. Extract Possible Conditions Cleanly
        possible_conditions = map_parameters_to_conditions(features)
        
        # 4. Grounded LLM Reasoning Engine
        reasoning = _reasoning_engine.analyze(features, risk_level, possible_conditions)
        
        # 5. Generate Confidence Reason
        confidence_reason = generate_confidence_reason(features, reasoning.get("trend_analysis", ""))
        
        # 6. Final Structured Output Formatting
        response = {
            "risk_level": risk_level,
            "confidence": confidence,
            "confidence_reason": confidence_reason,
            "trend_analysis": reasoning.get("trend_analysis", ""),
            "clinical_summary": reasoning.get("clinical_summary", ""),
            "llm_explanation": reasoning.get("llm_explanation", ""),
            "possible_conditions": possible_conditions,
            "recommendation": reasoning.get("recommendation", "")
        }
        
        return response
    except Exception as e:
        return {
            "risk_level": "Unknown",
            "confidence": 0.0,
            "confidence_reason": "Error generating confidence metric.",
            "trend_analysis": "Error computing trends",
            "clinical_summary": "Error during processing.",
            "llm_explanation": str(e),
            "possible_conditions": [],
            "recommendation": "System error occurred."
        }
