import json
import os
import sys

# Ensure imports work regardless of execution directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medical_model.pipeline import predict

def run_tests():
    print("--- Running Test 1: Full History, Contraindication trigger, Personalization ---")
    patient_data_1 = {
        "report_type": "lab",
        "personalization": {
            "age": 62,
            "gender": "male",
            "lifestyle": "sedentary"
        },
        "current_report": {
            "glucose": 65,            # Hypoglycemic
            "cholesterol": 240,       # High
            "hemoglobin": 14.0,       # Normal
            "blood_pressure_sys": 140, # High
            "blood_pressure_dia": 90,  # High
            "medication": "metformin and lisinopril" # Metformin contraindicated with glucose 65
        },
        "history": [
            {
                "date": "2025-01-01",
                "glucose": 150,
                "cholesterol": 200,
                "hemoglobin": 14.5,
                "diagnoses": ["Diabetes", "Hypertension"]
            },
            {
                "date": "2025-06-01",
                "glucose": 110,
                "cholesterol": 220,
                "hemoglobin": 14.2,
                "diagnoses": ["Diabetes", "Hypertension"]
            }
        ]
    }
    
    result_1 = predict(patient_data_1)
    print(json.dumps(result_1, indent=2))
    
    print("\n--- Running Test 2: Low Data Quality & Scan Formatting ---")
    patient_data_2 = {
        "report_type": "scan",
        "current_report": {
            "findings": "Large hyperdense mass found in anterior cavity. Slight inflammation."
        },
        "history": [] # No history = logic penalizes data_quality to 'low' and drops confidence
    }
    
    result_2 = predict(patient_data_2)
    print(json.dumps(result_2, indent=2))

if __name__ == "__main__":
    run_tests()
