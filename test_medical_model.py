import json
from medical_model.pipeline import predict

patient_data = {
    "current_report": {
        "glucose": 230,
        "cholesterol": 260,
        "hemoglobin": 9
    },
    "history": [
        {"glucose": 180, "cholesterol": 210},
        {"glucose": 200, "cholesterol": 230}
    ]
}

if __name__ == "__main__":
    result = predict(patient_data)
    print(json.dumps(result, indent=4))
