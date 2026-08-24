from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import src.meta_learners
import joblib
from pathlib import Path
import random

app = FastAPI(title="Uplift Modeling API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and models on startup
DATA_PATH = Path("data/simulation_observational.parquet")
MODELS_DIR = Path("models")

s_learner = None
dml_model = None
value_model = None
df = None
feature_cols = []
model_load_error = None

# We can dynamically calculate thresholds from the original dataset for the 2x2 grid
uplift_threshold = 0.0
value_threshold = 0.0

@app.on_event("startup")
def load_assets():
    global df, feature_cols, s_learner, dml_model, value_model, model_load_error, uplift_threshold, value_threshold
    try:
        df = pd.read_parquet(DATA_PATH)
        feature_cols = [c for c in df.columns if c not in ["treatment", "outcome", "tau_true", "propensity", "value_score"]]
        
        s_learner = joblib.load(MODELS_DIR / "s_learner.joblib")
        dml_model = joblib.load(MODELS_DIR / "doubleml_model.joblib")
        
        # Load new Value Model
        value_model = joblib.load(MODELS_DIR / "value_model.joblib")
        
        # Compute thresholds from historical dataset to align with Quadrant definitions
        X_hist = df[feature_cols].values
        historical_uplift = dml_model.effect(X_hist)
        historical_value = value_model.predict(X_hist)
        uplift_threshold = np.median(historical_uplift)
        value_threshold = np.median(historical_value)
        
    except Exception as e:
        model_load_error = str(e)
        print(f"Failed to load models: {e}")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Server is awake"}


@app.get("/api/random_user")
def get_random_user():
    # Pick a random user from the dataset
    if df is None:
        return {"error": "Data not loaded"}
    
    idx = random.randint(0, len(df) - 1)
    user_data = df.iloc[idx].to_dict()
    
    return {
        "index": idx,
        "features": user_data,
        "true_uplift": user_data.get("tau_true")
    }

class PredictRequest(BaseModel):
    features: dict

@app.post("/api/predict")
def predict_uplift(request: PredictRequest):
    if model_load_error:
        return {"error": model_load_error}
        
    # Construct X array in exact order of feature_cols
    x_dict = request.features
    x_array = np.array([[x_dict.get(c, 0.0) for c in feature_cols]])
    
    res = {}
    if s_learner is not None:
        tau_s = float(s_learner.predict(x_array)[0])
        res["s_learner_uplift"] = tau_s
        
    if dml_model is not None:
        tau_dml = float(dml_model.effect(x_array)[0])
        res["doubleml_uplift"] = tau_dml
        
    if value_model is not None:
        pred_val = float(value_model.predict(x_array)[0])
        res["predicted_value"] = pred_val
        
        # Quadrant segmentation
        if tau_dml >= uplift_threshold and pred_val >= value_threshold:
            segment = "Star Users"
            is_mismatch = False
        elif tau_dml >= uplift_threshold and pred_val < value_threshold:
            segment = "Mismatch"
            is_mismatch = True
        elif tau_dml < uplift_threshold and pred_val >= value_threshold:
            segment = "Sure Things"
            is_mismatch = False
        else:
            segment = "Lost Causes"
            is_mismatch = False
            
        res["segment"] = segment
        res["is_mismatch"] = is_mismatch
        
    return res

# Mount static files at the root
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
