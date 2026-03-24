import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train import load_data, NUMERIC_FEATURES, TEXT_FEATURE, TARGET

def evaluate_model(model_path='models/risk_model.pkl', encoder_path='models/label_encoder.pkl', data_path='data/synthetic_medical_reports.csv'):
    print(f"Loading model from {model_path}...")
    try:
        pipeline = joblib.load(model_path)
        le = joblib.load(encoder_path)
    except FileNotFoundError:
        print("Model or encoder file not found. Have you trained it yet?")
        return
        
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    
    X = df[NUMERIC_FEATURES + [TEXT_FEATURE]]
    y_true = df[TARGET]
    
    # Encode true labels to match the XGBoost output structure
    y_true_encoded = le.transform(y_true)
    
    print("Making predictions...")
    y_pred = pipeline.predict(X)
    
    # Metrics
    acc = accuracy_score(y_true_encoded, y_pred)
    f1 = f1_score(y_true_encoded, y_pred, average='weighted')
    report = classification_report(y_true_encoded, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_true_encoded, y_pred)
    
    print("--- EVALUATION METRICS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_p = os.path.join(base_dir, 'data', 'synthetic_medical_reports.csv')
    model_p = os.path.join(base_dir, 'models', 'risk_model.pkl')
    encoder_p = os.path.join(base_dir, 'models', 'label_encoder.pkl')
    evaluate_model(model_p, encoder_p, data_p)
