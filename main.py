from fastapi import FastAPI, Query, Path
from typing import List, Optional
import pandas as pd
import joblib
import os

app = FastAPI(title="Mission Score Predictor API")

OUTPUT_DIR = "output"
DATA_PATH = "Mockdata_Clean.csv"

feature_cols = [
    'R-test Score', 'PT Score', 'Year of Service', 'Deployment Service Year',
    'KPI from Supervisor', 'KPI from Peers', 'KPI from Subordinate',
    'Active Duty Day in One Year', 'Number of Mission Assigned', 'Number of Mission Succeed',
    'UN English Test Score', 'UN Knowledge Test Score',
    'Timeliness_normalize', 'Deployment_experience_score',
    'Language_skill_score', 'Rate_of_success_100_scale'
]

@app.get("/predict/mission")
def predict_mission(
    gender: str = Query(..., example="Female"),
    ranks: List[str] = Query(..., example=["พันตรี", "ร้อยตรี"]),
    mission_type: str = Query(..., example="UNMEM"),
    n_person: Optional[int] = Query(None, ge=1, description="Number of top persons to return"),
    short_detail: bool = Query(True, description="Return short detail only (id, name, rank, score)")
):
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return {"error": "CSV data file not found."}

    # Filter gender
    if gender != "All":
        df = df[df['Gender'] == gender]

    # Filter ranks
    df = df[df['Rank'].isin(ranks)]

    if df.empty:
        return {"message": "No records matched the filter criteria."}

    if mission_type == "All":
        # Load both models
        model1_path = os.path.join(OUTPUT_DIR, "UNMEM_model.pkl")
        model2_path = os.path.join(OUTPUT_DIR, "UNSO_model.pkl")
        if not os.path.exists(model1_path) or not os.path.exists(model2_path):
            return {"error": "One or both model files not found for 'All' mission type."}
        
        model1 = joblib.load(model1_path)
        model2 = joblib.load(model2_path)
        pred1 = model1.predict(df[feature_cols])
        pred2 = model2.predict(df[feature_cols])
        df["Predicted_Avg_Score"] = (pred1 + pred2) / 2
        sort_col = "Predicted_Avg_Score"
    else:
        # Single model case
        model_file = f"{mission_type}_model.pkl"
        model_path = os.path.join(OUTPUT_DIR, model_file)
        if not os.path.exists(model_path):
            return {"error": f"Model file not found: {model_file}"}
        model = joblib.load(model_path)
        predictions = model.predict(df[feature_cols])
        pred_col = f"Predicted_{mission_type}_Score"
        df[pred_col] = predictions
        sort_col = pred_col

    df_sorted = df.sort_values(by=sort_col, ascending=False)
    if n_person:
        df_sorted = df_sorted.head(n_person)

    result_col = sort_col if short_detail else None

    if short_detail:
        result_cols = ['id', 'Name', 'Rank', 'Gender', sort_col]
    else:
        result_cols = list(df_sorted.columns)

    results = df_sorted[result_cols].fillna("").to_dict(orient="records")

    return {
        "filter": {"gender": gender, "ranks": ranks, "mission_type": mission_type},
        "top_n": n_person if n_person else len(results),
        "short_detail": short_detail,
        "count": len(results),
        "results": results
    }


@app.get("/person/id/{person_id}")
def get_person_by_id(person_id: str = Path(..., example="U0005")):
    try:
        df = pd.read_csv(DATA_PATH)
        df = df[df['id']==person_id]
        return {
            "count": len(df),
            "results": df.fillna("").to_dict(orient="records")
        }
    except FileNotFoundError:
        return {"error": "CSV data file not found."}

@app.get("/person/all")
def get_all_persons():
    try:
        df = pd.read_csv(DATA_PATH)
        return {
            "count": len(df),
            "results": df.fillna("").to_dict(orient="records")
        }
    except FileNotFoundError:
        return {"error": "CSV data file not found."}
