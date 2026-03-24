import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_engine.rule_base import MEDICAL_RANGES

class RiskScorer:
    def __init__(self):
        self.ranges = MEDICAL_RANGES
        
    def score_parameters(self, extracted_params):
        """
        Takes a dict of extracted parameters and returns a risk score 
        and a list of 'danger' parameters.
        """
        risk_score = 0
        danger_parameters = []
        
        for param, value in extracted_params.items():
            if param in self.ranges:
                ref = self.ranges[param]
                
                # Special cases or severity jumps
                if param == "blood_glucose":
                    if value > 125 or value < 65:
                        risk_score += 2
                        danger_parameters.append((param, value, ref, "Critical Out of Range"))
                    elif value > ref["high"] or value < ref["low"]:
                        risk_score += 1
                        danger_parameters.append((param, value, ref, "Out of Range"))
                
                elif param == "cholesterol":
                    if value > 240:
                        risk_score += 2
                        danger_parameters.append((param, value, ref, "Critical Out of Range"))
                    elif value > ref["high"]:
                        risk_score += 1
                        danger_parameters.append((param, value, ref, "Out of Range"))

                elif param == "blood_pressure_sys":
                    if value > 140 or value < 80:
                        risk_score += 2
                        danger_parameters.append((param, value, ref, "Critical Out of Range"))
                    elif value > ref["high"] or value < ref["low"]:
                        risk_score += 1
                        danger_parameters.append((param, value, ref, "Out of Range"))
                
                elif param == "blood_pressure_dia":
                    if value > 90 or value < 50:
                        risk_score += 2
                        danger_parameters.append((param, value, ref, "Critical Out of Range"))
                    elif value > ref["high"] or value < ref["low"]:
                        risk_score += 1
                        danger_parameters.append((param, value, ref, "Out of Range"))
                
                else: # Standard scoring for everything else
                    if value > ref["high"] or value < ref["low"]:
                        risk_score += 1
                        danger_parameters.append((param, value, ref, "Out of Range"))
                        
        # Basic heuristic mapping corresponding to our synthetic data generator
        if risk_score >= 4:
            calculated_risk = "High"
        elif risk_score >= 2:
            calculated_risk = "Medium"
        else:
            calculated_risk = "Low"
            
        return calculated_risk, danger_parameters
