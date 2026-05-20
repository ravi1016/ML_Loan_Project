import os
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "loan_approval_dataset.csv"
ARTIFACTS_DIR = PROJECT_DIR / "model_artifacts"

# Model output files
CLASSIFIER_PATH = ARTIFACTS_DIR / "stage_1_classifier.pkl"
REGRESSOR_PATH = ARTIFACTS_DIR / "stage_2_regressor.pkl"

# Model hyperparameters
RANDOM_STATE = 42

# Column names
TARGET_COL = "loan_status"
REG_TARGET_COL = "loan_amount"
DROP_COLS_CLF = ["loan_status", "loan_amount", "loan_id"]
DROP_COLS_REG = ["loan_amount", "loan_status", "loan_id"]
