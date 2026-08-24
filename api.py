from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
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

df = None
feature_cols = []
s_learner = None
dml_model = None

@app.on_event("startup")
def load_assets():
    global df, feature_cols, s_learner, dml_model
    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ["treatment", "outcome", "tau_true", "propensity"]]
    
    print("Loading models...")
    try:
        s_learner = joblib.load(MODELS_DIR / "s_learner.joblib")
        dml_model = joblib.load(MODELS_DIR / "doubleml_model.joblib")
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Failed to load models: {e}")

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Server is awake"}

@app.get("/api/random_user")
def get_random_user():
    # Pick a random user from the dataset
    idx = random.randint(0, len(df) - 1)
    user_data = df.iloc[idx].to_dict()
    # Remove hidden/true causal data from the payload so we don't cheat
    for col in ["tau_true", "propensity", "outcome", "treatment"]:
        user_data.pop(col, None)
    return {"index": idx, "features": user_data}

class PredictRequest(BaseModel):
    features: dict

@app.post("/api/predict")
def predict_uplift(request: PredictRequest):
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
        
    return res

# Mount static files at the root
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
