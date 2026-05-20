import os
import sys
import pandas as pd

# Ensure current directory is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loan_approval.pipeline.prediction_pipeline import PredictionPipeline

def test_prediction():
    print("Initializing prediction pipeline...")
    pipeline = PredictionPipeline()
    
    # Check if models are loaded
    if pipeline.clf is None or pipeline.reg is None:
        print("Error: Models are not trained yet! Please run 'python train.py' first.")
        return
        
    sample = pd.DataFrame([{
        "no_of_dependents": 1,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 1200000,
        "loan_term": 12,
        "cibil_score": 820,
        "residential_assets_value": 2000000,
        "commercial_assets_value": 500000,
        "luxury_assets_value": 0,
        "bank_asset_value": 550000
    }])
    
    print("\nSample applicant data:")
    print(sample)
    
    print("\nRunning two-stage prediction...")
    result = pipeline.predict(sample)
    
    print("\nPrediction Results:")
    print("="*40)
    print(f"Approved: {result['approved']}")
    print(f"Approval Probability: {result['approval_probability']:.4f}")
    if result['approved']:
        print(f"Recommended Loan Amount: INR {result['recommended_loan_amount']:.2f}")
    print("="*40)

if __name__ == "__main__":
    test_prediction()
