import sys
import os
import argparse
import joblib
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr.ocr_engine import OCREngine
from preprocessing.text_cleaner import TextCleaner
from extraction.parser import ParameterParser
from risk_engine.severity_scorer import RiskScorer
from app.explainer import Explainer
from explainability.shap_explainer import SHAPExplainer
from medical_llm.inference import MedicalLLMInference

class MedicalAIAssistant:
    def __init__(self, model_path='models/risk_model.pkl', encoder_path='models/label_encoder.pkl', llm_path='models/medical_llm_weights.pth'):
        # Initialize modules
        self.ocr = OCREngine()
        self.cleaner = TextCleaner()
        self.parser = ParameterParser()
        self.scorer = RiskScorer()
        
        self.model = None
        self.encoder = None
        self.shap_explainer = None
        
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
            self.shap_explainer = SHAPExplainer(self.model)
            
        self.explainer = Explainer(self.shap_explainer)
        self.llm_generator = MedicalLLMInference(llm_path)

    def analyze_report(self, file_path):
        # 1. OCR / Text Extraction
        raw_text = self.ocr.extract_text(file_path)
        if not raw_text:
            return json.dumps({"error": "Failed to extract text."})

        # 2. Text Cleaning & Tokenization
        cleaned_text = self.cleaner.clean_text(raw_text)

        # 3. Parameter Extraction
        extracted_params = self.parser.extract_numerical_parameters(cleaned_text)

        # 4. Rule-based Risk Engine
        rule_risk, danger_params = self.scorer.score_parameters(extracted_params)

        # 5. ML Model Prediction
        ml_risk_str = "Unknown"
        ml_class_idx = None
        confidence = 0.0
        df_input = None
        
        if self.model and self.encoder:
            input_data = {
                'Blood_Glucose': extracted_params.get('blood_glucose', 0),
                'Cholesterol': extracted_params.get('cholesterol', 0),
                'Hemoglobin': extracted_params.get('hemoglobin', 0),
                'Blood_Pressure_Sys': extracted_params.get('blood_pressure_sys', 0),
                'Blood_Pressure_Dia': extracted_params.get('blood_pressure_dia', 0),
                'Platelet_Count': extracted_params.get('platelet_count', 0),
                'Heart_Rate': extracted_params.get('heart_rate', 0),
                'Report_Text': cleaned_text
            }
            df_input = pd.DataFrame([input_data])
            try:
                # Predict class index and decode
                ml_class_idx = self.model.predict(df_input)[0]
                ml_risk_str = self.encoder.inverse_transform([ml_class_idx])[0]
                
                # Get prediction probabilities for confidence
                probs = self.model.predict_proba(df_input)[0]
                confidence = float(probs[ml_class_idx])
            except Exception as e:
                pass

        # 6. Explainability output (Rule/SHAP) and Custom LLM 
        # SHAP explanation string (retained internal for potential verbose mode but omitting from structured JSON payload)
        explanation = self.explainer.generate_explanation(rule_risk, ml_risk_str, danger_params, df_input, ml_class_idx)
        
        # New Medical LLM Reasoning
        final_risk = ml_risk_str if ml_risk_str != "Unknown" else rule_risk
        llm_reasoning = self.llm_generator.generate_explanation(final_risk, danger_params, extracted_params)
        
        # 7. Format danger parameters for JSON
        danger_keys = [param[0] for param in danger_params]
        
        # The prompt specifically required these exact 4 keys in this structure
        final_output = {
            "risk_level": final_risk,
            "confidence": round(confidence, 2),
            "danger_parameters": danger_keys,
            "llm_reasoning": llm_reasoning
        }
        
        print(json.dumps(final_output, indent=4))
        return json.dumps(final_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Report Analyzer")
    parser.add_argument('--file', type=str, help='Path to the medical report image, pdf, or txt file')
    args = parser.parse_args()
    
    # We will use relative pathing to start this from medical_ai
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_loc = os.path.join(base_dir, 'models', 'risk_model.pkl')
    encoder_loc = os.path.join(base_dir, 'models', 'label_encoder.pkl')
    
    assistant = MedicalAIAssistant(model_loc, encoder_loc)
    
    if args.file:
        assistant.analyze_report(args.file)
    else:
        # Create a temp txt file for testing
        test_file = os.path.join(base_dir, 'data', 'temp_test.txt')
        os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("Patient's fasting blood glucose is 230 mg/dL. Total cholesterol is 260 mg/dL. Blood pressure 150/95 mmHg. Heart rate is 85. Hemoglobin is 9.0 g/dL. Platelets 250,000 mcL. Patient complains of feeling dizzy and tired.")
        assistant.analyze_report(test_file)
        os.remove(test_file)
