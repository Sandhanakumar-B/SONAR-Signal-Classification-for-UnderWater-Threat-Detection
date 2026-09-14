"""
tune.py
-------
Day 6 Execution Script: Hyperparameter Tuning, Threshold Optimization & Comprehensive Model Diagnostics.

Executes the comprehensive Day 6 machine learning optimization pipeline:
  1. Loads preprocessed datasets (data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv)
  2. Runs 5-Fold Stratified GridSearchCV hyperparameter tuning across all 4 classification models:
       - Support Vector Machine (C, gamma, kernel)
       - Random Forest (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
       - K-Nearest Neighbors (n_neighbors, weights, metric)
       - Logistic Regression (C, solver)
  3. Benchmarks Tuned models against Day 5 Baselines, quantifying performance deltas
  4. Optimizes Naval Threat Decision Thresholds:
       - Default (tau = 0.50)
       - Maximum F1-Score Operating Point
       - Safe High-Recall Zero-Miss Operating Point (targeting 100% mine detection)
       - Cost-Weighted Operating Point (5x penalty for lethal False Negatives)
  5. Computes Comprehensive Diagnostics:
       - Learning curves across sample size fractions (20% to 100%)
       - Acoustic frequency feature importances (identifying top discriminative sonar bands)
       - Probability calibration curves and Brier reliability scores
  6. Renders 5 publication-quality diagnostic plots in results/:
       - tuning_comparison.png
       - threshold_optimization.png
       - learning_curves.png
       - feature_importance.png
       - calibration_curves.png
  7. Persists tuning & diagnostic tables to results/ (tuning_metrics.csv, threshold_analysis.csv, tuning_and_diagnostics.json)
  8. Serializes all tuned models into models/ (tuned_svm.joblib, tuned_random_forest.joblib, tuned_knn.joblib, tuned_logistic_regression.joblib, best_tuned_model.joblib)
"""

import sys
from src.tuner import run_full_tuning


def main():
    try:
        # Run complete Day 6 hyperparameter tuning & diagnostics pipeline
        output = run_full_tuning(
            cv=5,
            random_state=42,
        )

        print("\n[INFO] All Day 6 hyperparameter tuning, threshold optimization, and diagnostic artifacts successfully completed.")
    except Exception as err:
        print(f"[ERROR] Failed to execute Day 6 tuning pipeline: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
