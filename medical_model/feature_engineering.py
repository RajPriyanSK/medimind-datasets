def time_weighted_trend(current_val, history_vals):
    if not history_vals:
        return current_val, "stable"
        
    decay_factor = 0.8
    weighted_sum = 0
    weight_total = 0
    
    for i, val in enumerate(history_vals):
        w = decay_factor ** i
        weighted_sum += val * w
        weight_total += w
        
    weighted_avg = weighted_sum / weight_total
    dev_val = current_val - weighted_avg
    
    threshold = 0.02 * weighted_avg if weighted_avg != 0 else 0
    rapid_threshold = 0.05 * weighted_avg if weighted_avg != 0 else 0
    
    if dev_val > rapid_threshold:
        trend = "rapidly increasing"
    elif dev_val > threshold:
        trend = "gradually increasing"
    elif dev_val < -threshold:
        trend = "decreasing"
    else:
        trend = "stable"
        
    return round(weighted_avg, 2), trend

def compute_features(patient_data: dict) -> dict:
    """
    Computes features based on current report and history.
    """
    current_report = patient_data.get("current_report", {})
    history = patient_data.get("history", [])
    report_type = patient_data.get("report_type", "unknown")
    
    # Sort history by date descending if dates exist to ensure newer records get higher weight
    try:
        sorted_history = sorted(history, key=lambda x: x.get("date", ""), reverse=True)
    except Exception:
        sorted_history = history
    
    features = {"report_type": report_type}
    
    for key, current_val in current_report.items():
        features[f"current_{key}"] = current_val
        
        history_vals = [past[key] for past in sorted_history if key in past]
        
        if isinstance(current_val, (int, float)) and not isinstance(current_val, bool):
            num_history = [v for v in history_vals if isinstance(v, (int, float))]
            if not num_history:
                features[f"avg_{key}"] = current_val
                features[f"weighted_avg_{key}"] = current_val
                features[f"trend_{key}"] = "stable"
                features[f"dev_{key}"] = 0.0
            else:
                avg_val = sum(num_history) / len(num_history)
                features[f"avg_{key}"] = round(avg_val, 2)
                
                weighted_avg, trend = time_weighted_trend(current_val, num_history)
                features[f"weighted_avg_{key}"] = weighted_avg
                features[f"trend_{key}"] = trend
                
                dev_val = current_val - weighted_avg
                features[f"dev_{key}"] = round(dev_val, 2)
        else:
            features[f"trend_{key}"] = "stable"
            
    return features
