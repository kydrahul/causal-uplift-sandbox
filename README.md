# Causal Inference Sandbox: Uplift Modeling for User Retention

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

* **ML & Data**: Python, `DoubleML`, `EconML`, `LightGBM`, `pandas`, `scikit-learn`
* **Backend API**: `FastAPI`, `Uvicorn`
* **Frontend**: `React`, `Vite`, `Tailwind CSS`, `Shadcn UI`

## 📦 Installation & Setup

### 1. Python Environment (Backend & ML)
Ensure you have Python 3.10+ installed.

```bash
# Install required Python packages
pip install -r requirements.txt
```

### 2. Node Environment (Frontend)
Ensure you have Node.js 18+ installed.

```bash
cd frontend
npm install
npm run build
cd ..
```
*Note: The FastAPI server is configured to serve the compiled frontend directly from `frontend/dist`.*

## 🏃‍♂️ Running the Project

### Step 1: Run the ML Pipeline (Optional)
If you want to re-train the models from scratch and generate new evaluation metrics:
```bash
python run_pipeline.py
```
This script will simulate data, train Meta-Learners and DoubleML models, evaluate AUUC/Qini curves, run refutation tests, and serialize the final models into the `models/` directory.

### Step 2: Start the Web Server
Launch the FastAPI server which serves both the ML API and the React frontend.
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to **http://localhost:8000** to view the sandbox!

## 🧪 Architecture Overview

1. **`src/`**: Core ML logic.
   - `data_simulation.py`: Generates a synthetic dataset with complex non-linear confounding to mimic a real-world subscription business.
   - `meta_learners.py`: Baseline uplift models.
   - `doubleml_estimator.py`: Advanced orthogonalized causal models.
   - `viz.py`: Matplotlib plotting logic for Qini and AUUC curves.
2. **`api.py`**: The FastAPI application serving POST inference routes.
3. **`frontend/`**: The Vite + React codebase for the interactive UI.
