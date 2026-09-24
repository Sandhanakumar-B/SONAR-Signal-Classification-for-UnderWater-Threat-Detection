"""
ensemble.py
-----------
Day 10 Execution Script: Deep Learning (Neural Networks - Multi-Layer Perceptron),
Advanced Ensemble Architectures (Stacking & Soft-Voting) & Multi-Model Benchmarking.

Executes the Day 10 pipeline:
  1. Loads preprocessed features and pre-existing tuned models.
  2. Constructs and trains a Deep Multi-Layer Perceptron (MLP) Neural Network (60 -> 64 -> 32 -> 1).
  3. Builds a calibrated Soft-Voting Ensemble (SVM, Random Forest, MLP, Logistic Regression).
  4. Builds a 2-Tier Stacking Classifier with Logistic Regression meta-learner and 5-fold CV.
  5. Computes comprehensive evaluation metrics on cross-validation and held-out test sets.
  6. Renders 4 publication-quality visualizations in results/:
       - neural_loss_curve.png
       - ensemble_comparison.png
       - ensemble_confusion_matrices.png
       - ensemble_roc_curves.png
  7. Persists serialized models to models/ and structured benchmark reports to results/.
"""

import sys
from src.ensemble import run_full_ensemble_pipeline


def main():
    try:
        print("=" * 78)
        print("   DAY 10: DEEP LEARNING & ADVANCED ENSEMBLE PIPELINE EXECUTION")
        print("=" * 78)

        report = run_full_ensemble_pipeline()

        print("\n" + "=" * 78)
        print("   DAY 10 PERFORMANCE BENCHMARK SUMMARY (Held-Out Test Set)")
        print("=" * 78)

        print(f"\n{'Model':<28} | {'Accuracy':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FN (Default)':<12} | {'FN (Safe)':<10}")
        print("-" * 88)

        for name, data in report["metrics"].items():
            acc = data["test_default"]["accuracy"] * 100
            rec = data["test_default"]["recall"] * 100
            f1 = data["test_default"]["f1_score"]
            fn_def = data["test_default"]["false_negatives"]
            fn_safe = data["test_safe"]["false_negatives"]
            print(f"{name:<28} | {acc:>8.2f}% | {rec:>8.2f}% | {f1:>10.4f} | {fn_def:>12d} | {fn_safe:>10d}")

        print("-" * 88)
        print("\n[SUCCESS] Day 10 Deep Learning & Ensemble pipeline completed successfully!")
        print("          All models serialized to models/ and visual diagnostics saved to results/.")

    except Exception as err:
        print(f"\n[ERROR] Failed to execute Day 10 ensemble pipeline: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
