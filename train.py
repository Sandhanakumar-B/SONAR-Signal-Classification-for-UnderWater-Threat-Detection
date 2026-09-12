"""
train.py
--------
Day 5 Execution Script: Supervised Model Training & Cross-Validation.
Executes the comprehensive Day 5 machine learning training pipeline:
  1. Loads preprocessed datasets (data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv)
  2. Evaluates 4 core supervised classification models using 5-Fold Stratified Cross-Validation:
       - Logistic Regression (Linear baseline)
       - K-Nearest Neighbors (KNN - Distance-based)
       - Support Vector Machine (SVM - RBF Kernel)
       - Random Forest (Ensemble Tree Bagging)
  3. Evaluates all trained models on held-out test data (42 unseen sonar signals)
  4. Audits threat detection performance (True Positives, False Negatives, Recall)
  5. Determines the champion classification model
  6. Generates and saves evaluation plots in results/:
       - model_comparison.png (Bar chart comparison across metrics)
       - confusion_matrices.png (2x2 grid for all models)
       - roc_curves.png (ROC curves & AUC comparison)
  7. Exports metrics summary to results/model_metrics.csv and results/model_evaluation_metrics.json
  8. Serializes trained models into models/ (logistic_regression.joblib, knn.joblib, svm.joblib, random_forest.joblib, best_model.joblib)
"""

import sys
from src.model_trainer import run_full_training


def main():
    try:
        # Run complete Day 5 model training & evaluation pipeline
        output = run_full_training(
            cv=5,
            random_state=42,
        )

        print("\n[INFO] All Day 5 model training, cross-validation, and evaluation artifacts successfully completed.")
    except Exception as err:
        print(f"[ERROR] Failed to execute Day 5 model training: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
