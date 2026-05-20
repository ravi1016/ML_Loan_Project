import os
import sys
from pathlib import Path

# Ensure the current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loan_approval.pipeline.training_pipeline import TrainingPipeline
from loan_approval.logger import logger

def train():
    try:
        logger.info("Initializing model training script...")
        pipeline = TrainingPipeline()
        metrics = pipeline.run_pipeline()
        
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print("Classifier Best Params:")
        print(metrics["clf_best_params"])
        print("\nClassifier Performance Report:")
        print(metrics["clf_report"])
        print("\nRegressor Best Params:")
        print(metrics["reg_best_params"])
        print("\nRegressor Performance Metrics:")
        for metric, val in metrics["reg_metrics"].items():
            print(f"  {metric}: {val:.4f}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        logger.exception(e)
        sys.exit(1)

if __name__ == "__main__":
    train()
