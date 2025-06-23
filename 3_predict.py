import pandas as pd
import joblib
import os

# ตั้งค่าพาธและ feature columns
OUTPUT_DIR = "output"

feature_cols = [
    'R-test Score', 'PT Score', 'Year of Service', 'Deployment Service Year',
    'KPI from Supervisor', 'KPI from Peers', 'KPI from Subordinate',
    'Active Duty Day in One Year', 'Number of Mission Assigned', 'Number of Mission Succeed',
    'UN English Test Score', 'UN Knowledge Test Score',
    'Timeliness_normalize', 'Deployment_experience_score',
    'Language_skill_score', 'Rate_of_success_100_scale'
]

def predict_mission(gender, ranks, mission_type):
    """
    Predict score from trained model based on gender, ranks, and mission type.
    
    Args:
        gender (str): Gender to filter (e.g., "Female")
        ranks (list): List of military ranks to include
        mission_type (str): One of ["UNMEM", "UNSO"]
    """
    df = pd.read_csv("Mockdata_Clean.csv")
    df = df[df['Gender'] == gender]
    df = df[df['Rank'].isin(ranks)]

    if df.empty:
        print("❌ No data matched filter criteria.")
        return

    model_file = f"{mission_type}_model.pkl"
    model_path = os.path.join(OUTPUT_DIR, model_file)

    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_file}")
        return

    model = joblib.load(model_path)
    predictions = model.predict(df[feature_cols])
    df[f"Predicted_{mission_type}_Score"] = predictions

    print(f"🔮 Predictions for {mission_type} mission:")
    print(df[['id', 'Name', 'Rank', 'Gender', f"Predicted_{mission_type}_Score"]])

if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    predict_mission(gender="Female", ranks=["พันตรี", "ร้อยตรี"], mission_type="UNMEM")
    predict_mission(gender="Male", ranks=["ร้อยโท", "พันโท"], mission_type="UNSO")
