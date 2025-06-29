import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error  # ✅ Use built-in RMSE function
)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

feature_cols = [
    'R-test Score', 'PT Score', 'Year of Service', 'Deployment Service Year',
    'KPI from Supervisor', 'KPI from Peers', 'KPI from Subordinate',
    'Active Duty Day in One Year', 'Number of Mission Assigned', 'Number of Mission Succeed',
    'UN English Test Score', 'UN Knowledge Test Score',
    'Timeliness_normalize', 'Deployment_experience_score',
    'Language_skill_score', 'Rate_of_success_100_scale'
]

models = {
    'LinearRegression': (LinearRegression(), {}),
    'Lasso': (Lasso(), {'alpha': [0.01, 0.1, 1.0]}),
    'Ridge': (Ridge(), {'alpha': [0.01, 0.1, 1.0]}),
    'ElasticNet': (ElasticNet(), {'alpha': [0.01, 0.1], 'l1_ratio': [0.2, 0.5]}),
    'RandomForest': (RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [5, 10]}),
    'AdaBoost': (AdaBoostRegressor(), {'n_estimators': [50, 100]}),
    'XGBoost': (XGBRegressor(), {'n_estimators': [100, 200], 'max_depth': [3, 6]})
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return {
        'Train_RMSE': root_mean_squared_error(y_train, y_train_pred),
        'Test_RMSE': root_mean_squared_error(y_test, y_test_pred),
        'Train_MAE': mean_absolute_error(y_train, y_train_pred),
        'Test_MAE': mean_absolute_error(y_test, y_test_pred),
        'Train_R2': r2_score(y_train, y_train_pred),
        'Test_R2': r2_score(y_test, y_test_pred)
    }

def train_and_save_best_model(df, target_col, name):
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model, best_score, best_name, best_params, best_eval = None, float("inf"), "", {}, {}

    for model_name, (model, param_grid) in models.items():
        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        else:
            grid = model  # no GridSearch

        try:
            if isinstance(grid, GridSearchCV):
                grid.fit(X_train, y_train)
                best_estimator = grid.best_estimator_
                params = grid.best_params_
            else:
                grid.fit(X_train, y_train)
                best_estimator = grid
                params = {}

            eval_metrics = evaluate_model(best_estimator, X_train, X_test, y_train, y_test)

            # Print model performance
            print(f"📊 Model: {model_name} | "
                  f"Train R²: {eval_metrics['Train_R2']:.4f} | "
                  f"Test R²: {eval_metrics['Test_R2']:.4f} | "
                  f"Test RMSE: {eval_metrics['Test_RMSE']:.4f}")

            # Save training log
            log_entry = {
                'Timestamp': datetime.now(),
                'Target': target_col,
                'Model': model_name,
                'BestParams': params,
                **eval_metrics
            }
            pd.DataFrame([log_entry]).to_csv(
                os.path.join(OUTPUT_DIR, "training_log.csv"),
                mode='a', index=False,
                header=not os.path.exists(os.path.join(OUTPUT_DIR, "training_log.csv"))
            )

            if eval_metrics['Test_RMSE'] < best_score:
                best_score = eval_metrics['Test_RMSE']
                best_model = best_estimator
                best_name = model_name
                best_params = params
                best_eval = eval_metrics

        except Exception as e:
            print(f"⚠️ {model_name} failed: {e}")

    # Save best model
    model_path = os.path.join(OUTPUT_DIR, f"{name}_model.pkl")
    joblib.dump(best_model, model_path)

    # Save best evaluation
    eval_df = pd.DataFrame([{
        'Target': target_col,
        'BestModel': best_name,
        'BestParams': best_params,
        **best_eval
    }])
    eval_df.to_csv(
        os.path.join(OUTPUT_DIR, "evaluation.csv"),
        mode='a', index=False,
        header=not os.path.exists(os.path.join(OUTPUT_DIR, "evaluation.csv"))
    )

    print(f"✅ Saved best model for {name}: {best_name} to {model_path}")

def predict_mission(gender, ranks, mission_type):
    df = pd.read_csv("Mockdata_Clean.csv")
    df = df[df['Gender'] == gender]
    df = df[df['Rank'].isin(ranks)]

    if df.empty:
        print("❌ No data matched filter criteria.")
        return

    model_file = f"{mission_type}_model.pkl"
    model_path = os.path.join(OUTPUT_DIR, model_file)

    if not os.path.exists(model_path):
        print(f"❌ Model for {mission_type} not found.")
        return

    model = joblib.load(model_path)
    predictions = model.predict(df[feature_cols])
    df[f"Predicted_{mission_type}_Score"] = predictions
    df[f"Actual_{mission_type}_Score"] = df[f"{mission_type}_Mission_success_score"]

    result_df = df[['id', 'Name', 'Rank', 'Gender',
                    f"Actual_{mission_type}_Score", f"Predicted_{mission_type}_Score"]]
    print(result_df.head())

    result_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{mission_type}_prediction_vs_actual.csv"),
        index=False
    )

# 🚀 Entry point
if __name__ == "__main__":
    df = pd.read_csv("Mockdata_Clean.csv")
    train_and_save_best_model(df, 'UNMEM_Mission_success_score', 'UNMEM')
    train_and_save_best_model(df, 'UNSO_Mission_success_score', 'UNSO')

    # 🔮 Example prediction
    predict_mission(gender="Female", ranks=["พันตรี", "ร้อยตรี"], mission_type="UNMEM")
