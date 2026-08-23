# Causal Inference Sandbox: Uplift Modeling for User Retention

[![Live Demo](https://img.shields.io/badge/Live_Demo-causal--uplift--sandbox.onrender.com-success?style=for-the-badge)](https://causal-uplift-sandbox.onrender.com)

An end-to-end Machine Learning pipeline and full-stack web application designed to demonstrate the power of **Causal Inference** and **Uplift Modeling**. 

While traditional predictive ML answers "Will this user churn?", Uplift Modeling answers a far more valuable business question: **"Will intervening (e.g., sending a promo code) *prevent* this user from churning?"**

This project utilizes SOTA causal estimators (Double Machine Learning) deployed via FastAPI and consumed by a sleek, modern React + Shadcn UI dashboard.

## 🚀 Key Features

* **Causal Machine Learning Pipeline**: 
  * Implements standard Meta-Learners (S, T, X, R) using `LightGBM`.
  * Implements **Double Machine Learning (DoubleML)** using `CausalForestDML` for robust, unbiased estimation of Conditional Average Treatment Effects (CATE).
  * Refutation Suite: Validates causal assumptions using Placebo Treatments, Random Common Causes, and Data Subsetting.
* **FastAPI Backend**: Serves trained `joblib` models with sub-20ms latency.
* **React + Shadcn UI Frontend**: A highly interactive, dark-mode SaaS dashboard that allows you to tweak user features and watch causal uplift predictions shift in real-time, complete with a live terminal log.

## 🛠️ Tech Stack

* **ML & Data**: Python, `DoubleML`, `LightGBM`, `pandas`, `scikit-learn`, `joblib`
* **Backend API**: `FastAPI`, `Uvicorn`
* **Frontend**: `React`, `Vite`, `Tailwind CSS`, `Shadcn UI`
* **Deployment**: Render (Full-Stack)

## 🧪 Architecture Overview

1. **`src/`**: Core ML logic.
   - `data_simulation.py`: Generates a synthetic dataset with complex non-linear confounding to mimic a real-world subscription business.
   - `meta_learners.py`: Baseline uplift models.
   - `doubleml_estimator.py`: Advanced orthogonalized causal models.
   - `viz.py`: Matplotlib plotting logic for Qini and AUUC curves.
2. **`api.py`**: The FastAPI application serving POST inference routes and mounting static frontend files.
3. **`frontend/`**: The Vite + React codebase for the highly interactive UI.

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
Launch the FastAPI server which serves both the ML API and the React frontend.
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
Then, open your browser and navigate to `http://localhost:8000`.
