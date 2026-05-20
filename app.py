import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure current directory is in Python path for package loading
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loan_approval.logger import logger
from loan_approval.pipeline.prediction_pipeline import PredictionPipeline
from loan_approval.pipeline.training_pipeline import TrainingPipeline

app = FastAPI(
    title="Loan Approval and Recommendation System",
    description="A modular two-stage machine learning system for loan approval classification and amount recommendation.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input data schema
class ApplicantSchema(BaseModel):
    no_of_dependents: int = Field(..., ge=0, description="Number of dependents of the applicant")
    education: str = Field(..., description="Education level of the applicant ('Graduate' or 'Not Graduate')")
    self_employed: str = Field(..., description="Employment status ('Yes' or 'No')")
    income_annum: int = Field(..., ge=0, description="Annual income of the applicant in INR")
    loan_term: int = Field(..., ge=1, description="Loan repayment term in months")
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL credit score")
    residential_assets_value: int = Field(..., ge=0, description="Value of residential assets owned")
    commercial_assets_value: int = Field(..., ge=0, description="Value of commercial assets owned")
    luxury_assets_value: int = Field(..., ge=0, description="Value of luxury assets owned")
    bank_asset_value: int = Field(..., ge=0, description="Value of bank assets/balance")

# Initialize pipelines
pred_pipeline = PredictionPipeline()
training_pipeline = TrainingPipeline()

# Serve static frontend files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    """Serves the main frontend page."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to the Loan Approval API. UI files are not generated yet."}

@app.post("/predict")
def predict_loan(applicant: ApplicantSchema):
    """
    Accepts applicant details and runs the two-stage prediction:
    1. Classification for approval / rejection.
    2. Regression for recommended loan amount (only if approved).
    """
    try:
        # Convert Pydantic model to dict, then to DataFrame
        data_dict = applicant.model_dump()
        applicant_df = pd.DataFrame([data_dict])
        
        # Execute prediction
        result = pred_pipeline.predict(applicant_df)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_models(background_tasks: BackgroundTasks):
    """
    Triggers model training in the background.
    """
    def run_training():
        try:
            training_pipeline.run_pipeline()
            # Reload pipelines after training is done
            pred_pipeline.load_models()
        except Exception as e:
            logger.error(f"Background training failed: {str(e)}")

    background_tasks.add_task(run_training)
    return {
        "status": "training_started",
        "message": "Model training has been triggered in the background. Check server logs for status."
    }

@app.get("/health")
def health_check():
    """API health status and model load check."""
    models_loaded = (pred_pipeline.clf is not None) and (pred_pipeline.reg is not None)
    return {
        "status": "healthy",
        "models_loaded": models_loaded
    }
