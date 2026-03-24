import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'medical_ai'))

from main import MedicalAIAssistant

assistant = MedicalAIAssistant('medical_ai/models/risk_model.pkl', 'medical_ai/models/label_encoder.pkl')
assistant.analyze_report('medical_ai/data/temp_test.txt')
