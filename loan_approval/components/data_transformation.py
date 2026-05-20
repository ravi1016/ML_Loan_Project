import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from loan_approval.logger import logger

class DataTransformation:
    def __init__(self):
        logger.info("Initializing DataTransformation component")

    def get_classification_preprocessor(self, num_cols: list, cat_cols: list) -> ColumnTransformer:
        """
        Builds and returns the preprocessor for Stage 1 (Classifier).
        """
        try:
            logger.info(f"Building classification preprocessor with num_cols={num_cols} and cat_cols={cat_cols}")
            
            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols)
            ])
            
            logger.info("Classification preprocessor created successfully")
            return preprocessor
        except Exception as e:
            logger.error(f"Error in creating classification preprocessor: {str(e)}")
            raise e

    def get_regression_preprocessor(self, num_cols_reg: list, cat_cols_reg: list) -> ColumnTransformer:
        """
        Builds and returns the preprocessor for Stage 2 (Regressor).
        """
        try:
            logger.info(f"Building regression preprocessor with num_cols={num_cols_reg} and cat_cols={cat_cols_reg}")
            
            # Adding SimpleImputer to make the regressor pipeline robust for inference
            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            reg_preprocessor = ColumnTransformer([
                ("num", numeric_transformer, num_cols_reg),
                ("cat", categorical_transformer, cat_cols_reg)
            ])
            
            logger.info("Regression preprocessor created successfully")
            return reg_preprocessor
        except Exception as e:
            logger.error(f"Error in creating regression preprocessor: {str(e)}")
            raise e
