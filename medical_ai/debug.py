import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import MedicalAIAssistant

assistant = MedicalAIAssistant('models/risk_model.pkl', 'models/label_encoder.pkl')

test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'temp_test.txt')
os.makedirs(os.path.dirname(test_file), exist_ok=True)
with open(test_file, 'w', encoding='utf-8') as f:
    f.write("Patient's fasting blood glucose is 230 mg/dL. Total cholesterol is 260 mg/dL. Blood pressure 150/95 mmHg. Heart rate is 85. Hemoglobin is 9.0 g/dL. Platelets 250,000 mcL. Patient complains of feeling dizzy and tired.")

try:
    assistant.analyze_report(test_file)
except Exception as e:
    with open('debug_trace.txt', 'w') as f:
        traceback.print_exc(file=f)
os.remove(test_file)
