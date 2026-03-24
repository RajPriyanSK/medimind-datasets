import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

class RiskModel:
    def __init__(self, model_path="medical_model/weights/xgboost_model.pkl", encoder_path="medical_model/weights/label_encoder.pkl"):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.encoder = None
        
        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            self.model = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)
            
    def predict(self, features: dict):
        """
        Receives dictionary of features and returns (risk_level, confidence).
        """
        if not self.model:
            return self._mock_predict(features)
            
        # Proper pipeline if model exists
        # Convert categorical trends to something suitable
        encoded_features = self._preprocess_features(features)
        df_input = pd.DataFrame([encoded_features])
        
        try:
            # Reorder columns to match training if specified
            # df_input = df_input[self.model.feature_names_in_]
            
            pred_idx = self.model.predict(df_input)[0]
            probs = self.model.predict_proba(df_input)[0]
            
            risk_level = self.encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            
            return risk_level, round(confidence, 2)
        except Exception as e:
            # Fallback in case of missing columns
            return self._mock_predict(features)
            
    def _preprocess_features(self, features: dict):
        """
        Encodes categorical trends to numeric for XGBoost without Pipeline if needed.
        """
        processed = features.copy()
        trend_map = {"increasing": 1, "stable": 0, "decreasing": -1}
        for k, v in list(processed.items()):
            if k.startswith("trend_"):
                processed[f"{k}_encoded"] = trend_map.get(v, 0)
                del processed[k]
        return processed
        
    def _mock_predict(self, features: dict):
        """
        Mock heuristic prediction if model weights aren't loaded yet.
        """
        risk_score = 0
        
        if features.get('current_glucose', 0) > 180:
            risk_score += 1
        if features.get('trend_glucose') == 'increasing':
            risk_score += 1
            
        if features.get('current_cholesterol', 0) > 240:
            risk_score += 1
        if features.get('trend_cholesterol') == 'increasing':
            risk_score += 1
            
        if risk_score >= 3:
            return "High Risk", 0.92
        elif risk_score >= 1:
            return "Medium Risk", 0.75
        else:
            return "Low Risk", 0.85
