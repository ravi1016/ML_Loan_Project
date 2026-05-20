import os
import pandas as pd
import numpy as np
import joblib
from loan_approval.logger import logger
from loan_approval.constants import CLASSIFIER_PATH, REGRESSOR_PATH

class PredictionPipeline:
    def __init__(self):
        self.clf = None
        self.reg = None
        self.load_models()

    def load_models(self):
        """Loads classifier and regressor models from artifacts directory."""
        try:
            if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
                logger.warning("Model files not found. They need to be trained first.")
                return False
            
            logger.info(f"Loading Stage 1 Classifier from {CLASSIFIER_PATH}")
            self.clf = joblib.load(CLASSIFIER_PATH)
            
            logger.info(f"Loading Stage 2 Regressor from {REGRESSOR_PATH}")
            self.reg = joblib.load(REGRESSOR_PATH)
            
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.clf = None
            self.reg = None
            return False

    def predict(self, applicant_df: pd.DataFrame) -> dict:
        """
        Executes the two-stage prediction logic:
        1. Classifies if applicant is approved and calculates probability.
        2. If approved, predicts recommended loan amount using regressor.
        """
        if self.clf is None or self.reg is None:
            # Try reloading in case they were recently trained
            if not self.load_models():
                raise FileNotFoundError("Models are not trained or could not be loaded.")

        try:
            logger.info("Running two-stage prediction for applicant data")
            
            # Ensure columns in applicant_df are stripped of whitespace
            applicant_df.columns = applicant_df.columns.str.strip()

            result = {}
            
            # Stage 1: Classification
            approve = self.clf.predict(applicant_df)[0]
            prob = self.clf.predict_proba(applicant_df)[0][1]
            
            result["approved"] = bool(approve)
            result["approval_probability"] = float(prob)
            result["recommended_loan_amount"] = None
            
            # Stage 2: Regression (only if approved)
            if approve == 1:
                predicted_amount = self.reg.predict(applicant_df)[0]
                result["recommended_loan_amount"] = float(predicted_amount)
                logger.info(f"Applicant APPROVED with prob={prob:.4f}. Recommended loan amount={predicted_amount:.2f}")
            else:
                logger.info(f"Applicant REJECTED with prob={prob:.4f}.")
                
            return result
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise e
