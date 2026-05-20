import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from loan_approval.logger import logger
from loan_approval.constants import RANDOM_STATE

class ModelTrainer:
    def __init__(self):
        logger.info("Initializing ModelTrainer component")

    def train_classifier(self, X_train, y_train, X_test, y_test, preprocessor) -> tuple:
        """
        Trains and tunes the Stage 1 Classifier using RandomizedSearchCV.
        """
        try:
            logger.info("Setting up Stage 1 Classifier pipeline...")
            clf_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(random_state=RANDOM_STATE))
            ])

            param_dist = {
                "model__n_estimators": [200, 300, 400, 500],
                "model__max_depth": [None, 5, 10, 15],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", None]
            }

            logger.info("Starting hyperparameter tuning for Stage 1 Classifier...")
            search = RandomizedSearchCV(
                clf_pipeline,
                param_dist,
                n_iter=20,
                cv=5,
                scoring="f1",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )

            search.fit(X_train, y_train)
            best_clf = search.best_estimator_
            
            logger.info(f"Best Classifier Params: {search.best_params_}")
            
            # Evaluation
            y_pred = best_clf.predict(X_test)
            clf_report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            logger.info("Stage 1 Classifier Evaluation:")
            logger.info(f"\n{clf_report}")
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
            
            return best_clf, search.best_params_, clf_report, conf_matrix
        except Exception as e:
            logger.error(f"Error in training classifier: {str(e)}")
            raise e

    def train_regressor(self, Xr_train, yr_train, Xr_test, yr_test, preprocessor) -> tuple:
        """
        Trains and tunes the Stage 2 Regressor using RandomizedSearchCV.
        """
        try:
            logger.info("Setting up Stage 2 Regressor pipeline...")
            reg_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(random_state=RANDOM_STATE))
            ])

            param_dist_reg = {
                "model__n_estimators": [200, 300, 400],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5, 10]
            }

            logger.info("Starting hyperparameter tuning for Stage 2 Regressor...")
            search_reg = RandomizedSearchCV(
                reg_pipeline,
                param_dist_reg,
                n_iter=15,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )

            search_reg.fit(Xr_train, yr_train)
            best_reg = search_reg.best_estimator_
            
            logger.info(f"Best Regressor Params: {search_reg.best_params_}")
            
            # Evaluation
            yr_pred = best_reg.predict(Xr_test)
            rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
            mae = mean_absolute_error(yr_test, yr_pred)
            r2 = r2_score(yr_test, yr_pred)
            
            logger.info("Stage 2 Regressor Evaluation:")
            logger.info(f"RMSE: {rmse:.4f}")
            logger.info(f"MAE: {mae:.4f}")
            logger.info(f"R2: {r2:.4f}")
            
            metrics = {
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            }
            
            return best_reg, search_reg.best_params_, metrics
        except Exception as e:
            logger.error(f"Error in training regressor: {str(e)}")
            raise e
