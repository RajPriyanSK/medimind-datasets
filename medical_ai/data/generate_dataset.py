import pandas as pd
import numpy as np
import random
import os

def generate_synthetic_data(num_samples=1000):
    data = []
    
    for i in range(num_samples):
        # Generate raw values
        glucose = round(random.uniform(60, 250), 1)
        cholesterol = round(random.uniform(100, 300), 1)
        hemoglobin = round(random.uniform(8.0, 19.0), 1)
        bp_sys = random.randint(80, 180)
        bp_dia = random.randint(50, 120)
        platelets = random.randint(100000, 600000)
        heart_rate = random.randint(50, 130)
        
        # Determine Severity / Risk Level
        risk_score = 0
        
        # Glucose Scoring
        if glucose > 125 or glucose < 65: risk_score += 2
        elif glucose > 100: risk_score += 1
        
        # Cholesterol
        if cholesterol > 240: risk_score += 2
        elif cholesterol > 200: risk_score += 1
            
        # BP
        if bp_sys > 140 or bp_dia > 90: risk_score += 2
        elif bp_sys > 120 or bp_dia > 80: risk_score += 1
            
        # Hemoglobin
        if hemoglobin < 11.0 or hemoglobin > 18.0: risk_score += 1
            
        # Platelets
        if platelets < 150000 or platelets > 450000: risk_score += 1
            
        # Heart Rate
        if heart_rate > 100 or heart_rate < 60: risk_score += 1
            
        # Assign risk level based on total score
        if risk_score >= 4:
            risk_level = "High"
        elif risk_score >= 2:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        # Generate Text Report
        templates = [
            f"Patient blood work indicates a fasting blood glucose level of {glucose} mg/dL and total cholesterol at {cholesterol} mg/dL. Hemoglobin is {hemoglobin} g/dL. Blood pressure recorded at {bp_sys}/{bp_dia} mmHg with a resting heart rate of {heart_rate} bpm. Platelet count evaluates to {platelets} /mcL.",
            
            f"Lab results: Glucose {glucose}. Cholesterol levels are {cholesterol} mg/dL. Patient's blood pressure is {bp_sys}/{bp_dia}. Hemoglobin: {hemoglobin}. HR: {heart_rate}. Platelets: {platelets}.",
            
            f"Routine checkup. Values extracted: Blood Glucose: {glucose} mg/dL, Total Cholesterol: {cholesterol} mg/dL, HB: {hemoglobin} g/dL, BP: {bp_sys}/{bp_dia}, Heart Rate: {heart_rate} bpm, Platelets: {platelets} mcL."
        ]
        
        report_text = random.choice(templates)
        
        data.append({
            "Patient_ID": i + 1,
            "Blood_Glucose": glucose,
            "Cholesterol": cholesterol,
            "Hemoglobin": hemoglobin,
            "Blood_Pressure_Sys": bp_sys,
            "Blood_Pressure_Dia": bp_dia,
            "Platelet_Count": platelets,
            "Heart_Rate": heart_rate,
            "Report_Text": report_text,
            "Risk_Level": risk_level
        })
        
    df = pd.DataFrame(data)
    os.makedirs('medical_ai/data', exist_ok=True)
    df.to_csv('medical_ai/data/synthetic_medical_reports.csv', index=False)
    print(f"Generated {num_samples} records in 'medical_ai/data/synthetic_medical_reports.csv'")

if __name__ == "__main__":
    generate_synthetic_data(1000)
