import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import joblib
import os
import sys

# Ensure we can import from medical_ai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_cleaner import TextCleaner

# Define features
NUMERIC_FEATURES = [
    'Blood_Glucose', 'Cholesterol', 'Hemoglobin', 
    'Blood_Pressure_Sys', 'Blood_Pressure_Dia', 
    'Platelet_Count', 'Heart_Rate'
]
TEXT_FEATURE = 'Report_Text'
TARGET = 'Risk_Level'

def load_data(data_path):
    df = pd.read_csv(data_path)
    # Basic data cleaning
    df = df.dropna(subset=[TARGET])
    return df

def train_model(data_path='data/synthetic_medical_reports.csv', model_save_path='models/risk_model.pkl', encoder_save_path='models/label_encoder.pkl'):
    print(f"Loading data from {data_path}...")
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"Data file not found at {data_path}. Please generate it first.")
        return None, None
        
    print(f"Dataset shape: {df.shape}")
    
    X = df[NUMERIC_FEATURES + [TEXT_FEATURE]]
    y = df[TARGET]
    
    # Preprocess text before TF-IDF using our custom text cleaner
    cleaner = TextCleaner()
    print("Cleaning text data for TF-IDF...")
    X.loc[:, TEXT_FEATURE] = X[TEXT_FEATURE].apply(cleaner.clean_text)
    
    # Label Encode Target for XGBoost
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('txt', TfidfVectorizer(max_features=500), TEXT_FEATURE)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='mlogloss', random_state=42))
    ])
    
    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
    
    search = RandomizedSearchCV(pipeline, param_grid, n_iter=5, cv=3, scoring='f1_weighted', random_state=42, n_jobs=-1)
    
    print("Training and tuning model...")
    search.fit(X_train, y_train)
    
    best_pipeline = search.best_estimator_
    print(f"Best parameters found: {search.best_params_}")
    
    print(f"Training completed. Saving model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_pipeline, model_save_path)
    joblib.dump(le, encoder_save_path)
    
    return best_pipeline, le, (X_test, y_test)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Correct paths relative to script placement
    data_path = os.path.join(base_dir, 'data', 'synthetic_medical_reports.csv')
    model_path = os.path.join(base_dir, 'models', 'risk_model.pkl')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoder.pkl')
    
    train_model(data_path, model_path, encoder_path)
