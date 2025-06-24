from fastapi import FastAPI, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import joblib
import os

app = FastAPI(title="Mission Score Predictor API")
# === FastAPI App ===
app = FastAPI(title="Bazi + User Management API")

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

def convert_to_card_format(person):
    try:
        physical = person.get("PT Score", 0)
        cognitive = round(sum([
            person.get("KPI from Supervisor", 0),
            person.get("KPI from Peers", 0),
            person.get("KPI from Subordinate", 0)
        ]) / 3, 2)
        performance = person.get("Rate_of_success_100_scale", 0)
        experience = person.get("Deployment_experience_score", 0)
        lang_score = min(person.get("Language_skill_score", 0) * 20, 100)

        badges = []
        if person.get("Military Course"):
            badges.append(person["Military Course"])
        if person.get("Expert Number Description"):
            badges.append(person["Expert Number Description"])

        language_tag = person.get("Extra Language Skills", "").strip()

        # Determine score column
        if "Predicted_UNMEM_Score" in person:
            predicted_score = person["Predicted_UNMEM_Score"]
        elif "Predicted_UNSO_Score" in person:
            predicted_score = person["Predicted_UNSO_Score"]
        elif "Predicted_Avg_Score" in person:
            predicted_score = person["Predicted_Avg_Score"]
        else:
            predicted_score = 0.0

        return {
            "name": person.get("Name", ""),
            "rank": person.get("Rank", ""),
            "radar": {
                "Physical": physical,
                "Cognitive": cognitive,
                "Performance": performance,
                "Experience": experience,
                "Language": lang_score
            },
            "badges": badges,
            "language_tag": language_tag,
            "predicted_score": round(predicted_score, 2)
        }
    except Exception as e:
        return {"error": f"Card conversion failed: {str(e)}"}


@app.get("/predict/mission")
def predict_mission(
    gender: str = Query(..., example="Female"),
    ranks: List[str] = Query(..., example=["พันตรี", "ร้อยตรี"]),
    mission_type: str = Query(..., example="UNMEM"),
    n_person: Optional[int] = Query(None, ge=1, description="Number of top persons to return"),
    short_detail: bool = Query(True, description="Return short detail only (card only or with full)")
):
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return {"error": "CSV data file not found."}

    if gender != "All":
        df = df[df['Gender'] == gender]
    df = df[df['Rank'].isin(ranks)]

    if df.empty:
        return {"message": "No records matched the filter criteria."}

    if mission_type == "All":
        mem_path = os.path.join(OUTPUT_DIR, "UNMEM_model.pkl")
        so_path = os.path.join(OUTPUT_DIR, "UNSO_model.pkl")
        if not os.path.exists(mem_path) or not os.path.exists(so_path):
            return {"error": "One or both model files for 'All' mission type not found."}
        model1 = joblib.load(mem_path)
        model2 = joblib.load(so_path)
        pred1 = model1.predict(df[feature_cols])
        pred2 = model2.predict(df[feature_cols])
        df["Predicted_Avg_Score"] = (pred1 + pred2) / 2
        score_col = "Predicted_Avg_Score"
    else:
        model_file = f"{mission_type}_model.pkl"
        model_path = os.path.join(OUTPUT_DIR, model_file)
        if not os.path.exists(model_path):
            return {"error": f"Model file not found: {model_file}"}
        model = joblib.load(model_path)
        pred = model.predict(df[feature_cols])
        score_col = f"Predicted_{mission_type}_Score"
        df[score_col] = pred

    df_sorted = df.sort_values(by=score_col, ascending=False)
    if n_person:
        df_sorted = df_sorted.head(n_person)

    # Convert each row to card + optionally full detail
    results = []
    for _, row in df_sorted.iterrows():
        row_dict = row.fillna("").to_dict()
        card = convert_to_card_format(row_dict)
        if short_detail:
            results.append({"card": card})
        else:
            results.append({"card": card, "detail": row_dict})

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
