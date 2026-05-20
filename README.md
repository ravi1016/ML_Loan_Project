Modular Loan Approval & Amount Prediction System

A production-style end-to-end Machine Learning project that predicts:

Loan Approval Status (Classification)
Recommended Loan Amount (Regression)

This project is built using:

Python
Scikit-learn
FastAPI
Random Forest Models
Modular ML Pipelines
Modern Responsive Frontend UI
🚀 Features
✅ Two-Stage Machine Learning Pipeline
Stage 1 — Loan Approval Prediction

Predicts whether a loan application should be:

Approved
Rejected

Using a tuned Random Forest Classifier.

Stage 2 — Recommended Loan Amount

If the application is approved, the system predicts the recommended loan amount using a tuned Random Forest Regressor.

🏗️ Project Architecture
loan_approval/
├── loan_approval/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── constants.py
│   └── logger.py
│
├── model_artifacts/
│   ├── stage_1_classifier.pkl
│   └── stage_2_regressor.pkl
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── app.py
├── train.py
├── requirements.txt
└── README.md

⚙️ Tech Stack
Category	Technology
Backend API	FastAPI
ML Framework	Scikit-learn
Models	Random Forest
Frontend	HTML, CSS, JavaScript
Serialization	Pickle
Hyperparameter Tuning	RandomizedSearchCV
Logging	Python Logging
Server	Uvicorn

📊 Machine Learning Workflow
Stage 1 — Classification Pipeline

Predicts loan approval status using:

Applicant income
CIBIL score
Assets
Education
Employment status
Loan term
Existing liabilities
Preprocessing
Missing value handling
Standard scaling
One-hot encoding
Model
Random Forest Classifier
Hyperparameter tuning using RandomizedSearchCV
Stage 2 — Regression Pipeline

If approved, predicts:

Recommended loan amount
Preprocessing
Standard scaling
One-hot encoding
Model
Random Forest Regressor
Hyperparameter tuning using RandomizedSearchCV
🎨 Frontend Features
Modern Glassmorphic UI
Dark futuristic design
Responsive layout
Smooth animations
Interactive sliders
Real-time validation
Dynamic UI Features
CIBIL Score Indicator
Score Range	Color
Poor	🔴 Red
Average	🟠 Orange
Excellent	🟢 Green
Prediction Dashboard

Displays:

Loan Approval Status
Approval Probability
Recommended Loan Amount
Visual Progress Gauge
🧠 Modular Design

The project follows clean software engineering principles:

Separation of concerns
Reusable pipelines
Component-based architecture
Scalable backend structure
Easy retraining and deployment
📦 Installation
1️⃣ Clone Repository
git clone <your-repository-url>
cd loan_approval
2️⃣ Create Virtual Environment

Using uv:

uv venv

Activate environment:

Windows
.venv\Scripts\activate
3️⃣ Install Dependencies
uv sync

OR using pip:

pip install -r requirements.txt
🏋️ Model Training

Run:

python train.py

This will:

Load dataset
Preprocess data
Train classifier
Train regressor
Evaluate models
Save .pkl artifacts
🧪 Run FastAPI Server

Start the API server:

uvicorn app:app --reload

If port 8000 is busy:

uvicorn app:app --reload --port 8001
🌐 Access Application
Frontend UI
http://127.0.0.1:8000
Swagger Documentation
http://127.0.0.1:8000/docs
🔌 API Endpoints
POST /predict

Predicts:

Approval status
Approval probability
Recommended loan amount
Example Request
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 750000,
  "loan_amount": 1500000,
  "loan_term": 12,
  "cibil_score": 780,
  "residential_assets_value": 500000,
  "commercial_assets_value": 300000,
  "luxury_assets_value": 100000,
  "bank_asset_value": 400000
}
Example Response
{
  "approved": true,
  "approval_probability": 0.91,
  "recommended_loan_amount": 1640000
}
🧾 Model Evaluation Metrics
Classification Metrics
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Regression Metrics
RMSE
MAE
R² Score
📈 Example Scenarios
❌ Rejected Applicant
Low CIBIL score
High requested loan amount
Low annual income

Result:

{
  "approved": false,
  "approval_probability": 0.18
}
✅ Approved Applicant
High CIBIL score
Stable income
Good assets

Result:

{
  "approved": true,
  "approval_probability": 0.93,
  "recommended_loan_amount": 2400000
}
🔒 Logging

Custom logging tracks:

Training pipeline execution
Model loading
Prediction requests
Errors and exceptions
🚀 Future Improvements
Docker support
CI/CD pipelines
Cloud deployment
Authentication system
Database integration
Model monitoring
Explainable AI (SHAP)
Real-time analytics dashboard
🛠️ Development Commands
Train Models
python train.py
Run Backend
uvicorn app:app --reload
Run On Different Port
uvicorn app:app --reload --port 8001