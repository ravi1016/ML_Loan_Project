import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from loan_approval.logger import logger
from loan_approval.constants import (
    DATASET_PATH,
    ARTIFACTS_DIR,
    CLASSIFIER_PATH,
    REGRESSOR_PATH,
    RANDOM_STATE,
    TARGET_COL,
    REG_TARGET_COL,
    DROP_COLS_CLF,
    DROP_COLS_REG
)
from loan_approval.components.data_transformation import DataTransformation
from loan_approval.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logger.info("=========================================")
            logger.info("Starting Training Pipeline Execution")
            logger.info("=========================================")

            # 1. Load data
            if not DATASET_PATH.exists():
                raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
            
            logger.info(f"Loading dataset from: {DATASET_PATH}")
            df = pd.read_csv(DATASET_PATH)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            logger.info(f"Dataset shape: {df.shape}")

            # 2. Stage 1 Prep
            logger.info("Preparing data for Stage 1 (Classification)")
            df[TARGET_COL] = df[TARGET_COL].str.strip().str.lower()
            df[TARGET_COL] = (df[TARGET_COL] == "approved").astype(int)

            X_clf = df.drop(columns=DROP_COLS_CLF)
            y_clf = df[TARGET_COL]

            num_cols_clf = X_clf.select_dtypes(include=np.number).columns.tolist()
            cat_cols_clf = X_clf.select_dtypes(include="object").columns.tolist()

            X_train, X_test, y_train, y_test = train_test_split(
                X_clf, y_clf,
                test_size=0.2,
                stratify=y_clf,
                random_state=RANDOM_STATE
            )

            # Preprocessing Stage 1
            clf_preprocessor = self.data_transformation.get_classification_preprocessor(
                num_cols=num_cols_clf, cat_cols=cat_cols_clf
            )

            # Train Stage 1
            best_clf, best_clf_params, clf_report, conf_matrix = self.model_trainer.train_classifier(
                X_train, y_train, X_test, y_test, clf_preprocessor
            )

            # 3. Stage 2 Prep
            logger.info("Preparing data for Stage 2 (Regression)")
            approved_df = df[df[TARGET_COL] == 1].copy()
            
            X_reg = approved_df.drop(columns=DROP_COLS_REG)
            y_reg = approved_df[REG_TARGET_COL]

            num_cols_reg = X_reg.select_dtypes(include=np.number).columns.tolist()
            cat_cols_reg = X_reg.select_dtypes(include="object").columns.tolist()

            Xr_train, Xr_test, yr_train, yr_test = train_test_split(
                X_reg, y_reg,
                test_size=0.2,
                random_state=RANDOM_STATE
            )

            # Preprocessing Stage 2
            reg_preprocessor = self.data_transformation.get_regression_preprocessor(
                num_cols_reg=num_cols_reg, cat_cols_reg=cat_cols_reg
            )

            # Train Stage 2
            best_reg, best_reg_params, reg_metrics = self.model_trainer.train_regressor(
                Xr_train, yr_train, Xr_test, yr_test, reg_preprocessor
            )

            # 4. Save Artifacts
            logger.info("Saving trained models to disk...")
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(best_clf, CLASSIFIER_PATH)
            joblib.dump(best_reg, REGRESSOR_PATH)
            
            logger.info(f"Stage 1 Classifier saved to: {CLASSIFIER_PATH}")
            logger.info(f"Stage 2 Regressor saved to: {REGRESSOR_PATH}")
            logger.info("=========================================")
            logger.info("Training Pipeline Execution Completed Successfully")
            logger.info("=========================================")

            return {
                "clf_best_params": best_clf_params,
                "clf_report": clf_report,
                "conf_matrix": conf_matrix.tolist(),
                "reg_best_params": best_reg_params,
                "reg_metrics": reg_metrics
            }
        except Exception as e:
            logger.error("Error running the training pipeline")
            logger.exception(e)
            raise e

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
