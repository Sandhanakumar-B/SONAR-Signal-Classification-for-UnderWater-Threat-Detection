"""
explain.py
----------
Day 8 Execution Script: Model Explainability, SHAP Interpretability & Error Analysis.

Executes the Day 8 explainability pipeline:
  1. Loads unseen test samples and the tuned champion SVM classification model
  2. Calculates test-set Permutation Feature Importance (F1 degradation on shuffle)
  3. Computes SHAP (SHapley Additive exPlanations) values via model-agnostic KernelExplainer
  4. Profiles decision boundary and misclassification cases under safe threshold (tau = 0.43)
  5. Renders 5 publication-quality interpretability visualisations to results/:
       - permutation_importance.png (Top 15 most sensitive frequencies)
       - shap_summary.png (Mean |SHAP| global importance)
       - shap_beeswarm.png (Beeswarm directional impact)
       - misclassification_analysis.png (Spectral profiles and safety margin scatter)
       - local_prediction_waterfall.png (Instance explanations for high-confidence predictions)
  6. Exports structured findings to results/explainability_report.json
"""

import sys
from src.explainer import run_full_explainability


def main():
    try:
        output = run_full_explainability(
            nsamples=100,
            background_size=40,
            random_state=42
        )

        print("\n" + "=" * 76)
        print("   DAY 8 EXPLAINABILITY SUMMARY")
        print("=" * 76)

        print("\n[*] Top 5 Most Influential Acoustic Frequencies (Permutation F1-Score):")
        for rank, rec in enumerate(output["permutation"]["records"][:5], start=1):
            print(f"    {rank}. {rec['feature']} -> Mean Delta F1: {rec['mean_importance']:.4f} (+/- {rec['std_importance']:.4f})")

        print("\n[*] Top 5 Most Influential Acoustic Frequencies (Global Mean |SHAP|):")
        for rank, rec in enumerate(output["shap"]["feature_shap_rank"][:5], start=1):
            print(f"    {rank}. {rec['feature']} -> Mean |SHAP|: {rec['mean_abs_shap']:.4f}")

        mis = output["misclassifications"]
        print(f"\n[*] Operational Safety Audit on Held-Out Test Set (Threshold = {mis['threshold']:.2f}):")
        print(f"    - True Positives  (Mines correctly intercepted) : {len(mis['tp_indices'])} / 22")
        print(f"    - False Negatives (Lethal missed mines)          : {len(mis['fn_indices'])} (Target: 0)")
        print(f"    - False Positives (False alarms on rocks)        : {len(mis['fp_indices'])} / 20")
        print(f"    - Zero-Miss Safety Status                         : {'PASSED [Zero Missed Mines]' if len(mis['fn_indices']) == 0 else 'WARNING [Missed Mines Detected]'}")

        print(f"\n[INFO] All Day 8 explainability plots and reports generated successfully.")
    except Exception as err:
        print(f"[ERROR] Failed to execute Day 8 explainability pipeline: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
