def compute_features(patient_data: dict) -> dict:
    """
    Computes features based on current report and history.
    """
    current_report = patient_data.get("current_report", {})
    history = patient_data.get("history", [])
    
    features = {}
    
    for key, current_val in current_report.items():
        # Current feature
        features[f"current_{key}"] = current_val
        
        # Historical values
        history_vals = [past[key] for past in history if key in past]
        
        if not history_vals:
            features[f"avg_{key}"] = current_val
            features[f"trend_{key}"] = "stable"
            features[f"dev_{key}"] = 0.0
        else:
            avg_val = sum(history_vals) / len(history_vals)
            features[f"avg_{key}"] = round(avg_val, 2)
            
            dev_val = current_val - avg_val
            features[f"dev_{key}"] = round(dev_val, 2)
            
            # Trend calculation (using 2% threshold for stability)
            threshold = 0.02 * avg_val if avg_val != 0 else 0
            if dev_val > threshold:
                features[f"trend_{key}"] = "increasing"
            elif dev_val < -threshold:
                features[f"trend_{key}"] = "decreasing"
            else:
                features[f"trend_{key}"] = "stable"
                
    return features
