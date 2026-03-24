import shap
import pandas as pd
import numpy as np
import xgboost as xgb

class SHAPExplainer:
    def __init__(self, pipeline):
        """
        Initializes the SHAP explainer with a trained scikit-learn Pipeline
        that contains a 'preprocessor' and a 'classifier' (XGBoost).
        """
        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps['preprocessor']
        self.model = pipeline.named_steps['classifier']
        self.booster = self.model.get_booster()
        
    def _get_feature_names(self):
        """Extracts feature names from the ColumnTransformer."""
        feature_names = []
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'drop':
                continue
            if name == 'remainder' and transformer == 'drop':
                continue
            if hasattr(transformer, 'get_feature_names_out'):
                try:
                    names = transformer.get_feature_names_out(features)
                    # Clean up the names (often they are prefixed with the transformer name)
                    names = [n.split('__')[-1] for n in names]
                    feature_names.extend(names)
                except Exception:
                    feature_names.extend([f"{name}_{i}" for i in range(len(features))])
            else:
                feature_names.extend([f"{name}_{f}" for f in features])
        return np.array(feature_names)

    def explain_prediction(self, input_df, target_class_idx):
        """
        Calculates SHAP values for a single prediction and returns the top risk factors.
        
        Args:
            input_df: DataFrame containing the input features for a single patient.
            target_class_idx: The integer class index predicted by the model (e.g., finding the risk level).
            
        Returns:
            list of tuples: [(feature_name, shap_value), ...] sorted by top absolute importance.
        """
        # Transform the input using the pipeline's preprocessor
        X_transformed = self.preprocessor.transform(input_df)
        
        # If it's a sparse matrix from TF-IDF, convert to dense for SHAP
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
            
        # Bypass shap.TreeExplainer due to string-conversion bug in XGBoost > 2.0 with base_score
        # Calculate SHAP values natively using xgboost pred_contribs
        dmatrix = xgb.DMatrix(X_transformed)
        shap_values = self.booster.predict(dmatrix, pred_contribs=True)
        
        # shap_values shape for multiclass: (num_samples, num_classes, num_features + 1 bias) 
        # or list of arrays (one per class).
        target_shap_values = None
        if isinstance(shap_values, list):
            target_shap_values = shap_values[target_class_idx][0]
        else:
            if len(shap_values.shape) == 3:
                # Shape (samples, classes, features + 1)
                target_shap_values = shap_values[0, target_class_idx, :-1]
            elif len(shap_values.shape) == 2:
                # Shape (samples, features + 1)
                target_shap_values = shap_values[0, :-1]
            else:
                target_shap_values = np.array(shap_values[:-1]).flatten()
                
        feature_names = self._get_feature_names()
        
        # Ensure target_shap_values is 1D
        target_shap_values = np.ravel(target_shap_values)
        
        # Pair feature names with their SHAP values safely
        feature_importance = [(feature_names[i], float(target_shap_values[i])) for i in range(min(len(feature_names), len(target_shap_values)))]
        
        # Sort by absolute SHAP value (impact magnitude) descending
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return top 5 influential features
        return feature_importance[:5]

