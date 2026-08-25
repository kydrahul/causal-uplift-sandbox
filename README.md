# Causal Inference Sandbox: Uplift Modeling for User Retention

[![Live Demo](https://img.shields.io/badge/Live_Demo-causal--uplift--sandbox.onrender.com-success?style=for-the-badge)](https://causal-uplift-sandbox.onrender.com)

An end-to-end Machine Learning pipeline and full-stack web application designed to demonstrate the power of **Causal Inference** and **Uplift Modeling**. 

While traditional predictive ML answers "Will this user churn?", Uplift Modeling answers a far more valuable business question: **"Will intervening (e.g., sending a push notification) *prevent* this user from churning?"**

This project utilizes SOTA causal estimators (Double Machine Learning) alongside standard predictive models to identify the **Persuadables**—users who only return *because* they received a notification—and avoids wasting money on "Sure Things" and "Lost Causes".

## 🚀 Key Features

* **Causal Machine Learning Pipeline**: 
  * Implements standard Meta-Learners (S, T, X, R) using `LightGBM`.
  * Implements **Double Machine Learning (DoubleML)** using `EconML`'s `CausalForestDML` for robust, unbiased estimation of True Causal Uplift (CATE).
  * Runs mathematically rigorous **Refutation Tests** to ensure stability against unobserved confounders (Placebo, Fake Confounders, Subsetting).
* **Value-Aware Segmentation**: 
  * Cross-references the causal uplift score against a **Lifetime Value Model (LGBM)** to predict not just *if* we can save a user, but if they are *worth* saving.
* **FastAPI Backend**: Serves trained `joblib` models and calculates complex causal inference predictions in sub-100ms.
* **React + Shadcn UI Frontend**: A stunning, highly interactive dark-mode SaaS dashboard. Features interactive tooltips, a live terminal logging window, and real-time inference tracking.

## 🛠️ Tech Stack

* **ML & Data**: Python, `EconML`, `LightGBM`, `pandas`, `scikit-learn`, `joblib`
* **Backend API**: `FastAPI`, `Uvicorn`
* **Frontend**: `React`, `Vite`, `Tailwind CSS`, `Shadcn UI`
* **Deployment**: Render (Full-Stack Monolith)

## 🧪 Architecture Overview

1. **`src/`**: Core ML and Data logic.
   - `simulate_rct.py`: Generates the core synthetic dataset containing both observational (biased) data and Randomized Control Trial (RCT) data, complete with complex non-linear confounders.
   - `feature_engineering.py`: Processes and cleans behavior profiles (e.g., Genre Entropy, Avg Rating) for modeling.
   - `meta_learners.py`: Trains the baseline standard uplift models.
   - `doubleml_estimator.py`: Trains the advanced orthogonalized causal model.
   - `value_model.py`: Trains the lifetime value prediction model used in the quadrant analysis.
   - `segment_analysis.py` & `viz.py`: Generates metrics, plots (Qini, AUUC), and Refutation Tests.
2. **`api.py`**: The FastAPI application serving POST inference routes and mounting static frontend files.
3. **`frontend/`**: The Vite + React codebase for the interactive user interface.
4. **`run_pipeline.py`**: The main execution script. Run this to generate the dataset, train all models from scratch, and output the metrics into the `results/` folder.

## 💻 Running Locally

If you wish to run the backend and explore the ML pipeline on your own machine:

### 1. Python Environment (Backend & ML)
Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

*(Optional)* If you want to re-train the models from scratch and generate new evaluation metrics:
```bash
python run_pipeline.py
```

### 2. Node Environment (Frontend)
Ensure you have Node.js 18+ installed.

```bash
cd frontend
npm install
npm run build
cd ..
```

### 3. Start the Web Server
Launch the FastAPI server which serves both the ML API and the React frontend on the same port.
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Then, open your browser and navigate to `http://localhost:8000`.
