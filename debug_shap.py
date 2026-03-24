import sys
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
sys.path.append(os.path.join(os.getcwd(), 'medical_ai'))

model = joblib.load('medical_ai/models/risk_model.pkl')
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

input_data = pd.DataFrame([{
    'Blood_Glucose': 230,
    'Cholesterol': 260,
    'Hemoglobin': 9.0,
    'Blood_Pressure_Sys': 150,
    'Blood_Pressure_Dia': 95,
    'Platelet_Count': 250000,
    'Heart_Rate': 85,
    'Report_Text': "Patient's fasting blood glucose is 230."
}])

try:
    X_trans = preprocessor.transform(input_data)
    if hasattr(X_trans, 'toarray'): X_trans = X_trans.toarray()

    booster = classifier.get_booster()
    dmatrix = xgb.DMatrix(X_trans)
    shap_v = booster.predict(dmatrix, pred_contribs=True)

    print("SHAP VALUES TYPE:", type(shap_v))
    print("ARRAY SHAPE:", shap_v.shape)
        
    print("FIRST ELEMENT:", shap_v[0])
except Exception as e:
    with open('error_log.txt', 'w') as f:
        import traceback
        traceback.print_exc(file=f)
