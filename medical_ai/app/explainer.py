class Explainer:
    def __init__(self, shap_explainer=None):
        self.shap_explainer = shap_explainer
        
    def generate_explanation(self, calculated_risk, ml_risk, danger_parameters, input_df=None, ml_class_idx=None):
        """
        Combines rule-based danger parameters and ML risk to explain the prediction.
        """
        explanation = []
        explanation.append(f"Model Predicted Risk: {ml_risk}")
        explanation.append(f"Rule-Based Calculated Risk: {calculated_risk}")
        
        if calculated_risk != ml_risk:
            explanation.append("Note: There is a discrepancy between the strict rule-based heuristic and the ML model prediction. The ML model takes into account complex non-linear interactions and text features.")
            
        if self.shap_explainer and input_df is not None and ml_class_idx is not None:
            try:
                top_features = self.shap_explainer.explain_prediction(input_df, ml_class_idx)
                explanation.append("\nKey Risk Factors (SHAP Feature Importance):")
                for feature, impact in top_features:
                    direction = "increased" if impact > 0 else "decreased"
                    explanation.append(f"- {feature}: {direction} risk (SHAP value: {impact:.4f})")
            except Exception as e:
                explanation.append(f"\nCould not calculate SHAP values: {e}")
                
        if not danger_parameters:
            explanation.append("\nAll extracted vital parameters are within normal reference ranges.")
        else:
            explanation.append("\nThe following parameters are out of normal reference ranges:")
            for param, value, ref, severity in danger_parameters:
                explanation.append(f"- {ref['name']}: {value} {ref['unit']} (Normal is {ref['low']}-{ref['high']} {ref['unit']}) -> {severity}")
                
        return "\n".join(explanation)

if __name__ == "__main__":
    ex = Explainer()
    print(ex.generate_explanation("High", "High", [("blood_glucose", 130, {"low": 70, "high": 100, "name": "Blood Glucose", "unit": "mg/dL"}, "Critical Out of Range")]))
