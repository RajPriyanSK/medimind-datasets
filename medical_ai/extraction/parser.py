import re

class ParameterParser:
    def __init__(self):
        # Maps canonical param keys to possible text occurrences 
        self.param_patterns = {
            "blood_glucose": r"(?:glucose|fasting blood glucose|fbg|fbs).*?(\d+(?:\.\d+)?)",
            "cholesterol": r"(?:cholesterol|total cholesterol|tc).*?(\d+(?:\.\d+)?)",
            "hemoglobin": r"(?:hemoglobin|hb|hgb).*?(\d+(?:\.\d+)?)",
            "blood_pressure": r"(?:blood pressure|bp).*?(\d{2,3})\s*[/|\|\\]\s*(\d{2,3})",
            "platelet_count": r"(?:platelet|platelet count|plt).*?(\d+(?:,\d+)?(?:\.\d+)?)",
            "heart_rate": r"(?:heart rate|hr|pulse).*?(\d{2,3})"
        }
    
    def extract_numerical_parameters(self, text):
        """
        Uses Regex to find the numeric values for medical parameters in the raw text.
        Returns a dictionary mapping parameter names to extracted floats.
        """
        text = text.lower()
        extracted = {}
        
        for key, pattern in self.param_patterns.items():
            match = re.search(pattern, text)
            if match:
                if key == "blood_pressure":
                    # Special case: Sys / Dia tuple capture
                    sys_v = float(match.group(1))
                    dia_v = float(match.group(2))
                    extracted["blood_pressure_sys"] = sys_v
                    extracted["blood_pressure_dia"] = dia_v
                else:
                    try:
                        val_str = match.group(1).replace(",", "")
                        extracted[key] = float(val_str)
                    except ValueError:
                        pass
        return extracted

if __name__ == '__main__':
    parser = ParameterParser()
    sample_text = "Routine checkup. Values extracted: Blood Glucose: 110.5 mg/dL, Total Cholesterol: 220, HB: 14.2 g/dL, BP: 145/95, Heart Rate: 85 bpm, Platelets: 250,000 mcL."
    print("Extracted:", parser.extract_numerical_parameters(sample_text))
