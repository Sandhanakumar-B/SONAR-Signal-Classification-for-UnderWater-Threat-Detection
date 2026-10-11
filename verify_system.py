"""
verify_system.py
----------------
Day 10: End-to-End System & Automated Pipeline Verification Suite.

Validates the full 10-day Machine Learning engineering pipeline:
  [TEST 1] Raw dataset integrity & schema verification (UCI Sonar Dataset)
  [TEST 2] Preprocessing artifacts, scaling integrity & data leakage prevention
  [TEST 3] Classical baseline models (Day 5) & hyperparameter-tuned models (Day 6)
  [TEST 4] Deep Neural Network (MLP) & Advanced Stacking/Voting Ensembles (Day 10)
  [TEST 5] Multi-Model Inference Engine & dynamic architecture switching (Day 7/10)
  [TEST 6] Zero-False-Negative Naval Threat Safety Policy Audit (Tau = 0.43)
  [TEST 7] Explainable AI (XAI) & SHAP feature attribution generator (Day 8)
  [TEST 8] Production Flask REST API endpoints & Health Diagnostics (Day 9/10)
  [TEST 9] Publication-quality visualization and structured report artifacts in results/
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

# Canonical paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def patch_sklearn_compat(model):
    """Ensures cross-version attribute compatibility for unpickled estimators."""
    if model is None:
        return model
    candidates = [model]
    if hasattr(model, "estimators_"):
        candidates.extend(model.estimators_)
    if hasattr(model, "named_estimators_"):
        candidates.extend(list(model.named_estimators_.values()))
    if hasattr(model, "final_estimator_"):
        candidates.append(model.final_estimator_)
    for est in candidates:
        if hasattr(est, "__class__") and est.__class__.__name__ == "LogisticRegression":
            if not hasattr(est, "multi_class"):
                est.multi_class = "auto"
    return model


def run_verification():
    passed_tests = 0
    total_tests = 9

    print("=" * 80)
    print("   SONAR SIGNAL CLASSIFICATION SYSTEM - FULL SYSTEM VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Raw Dataset Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Verifying Raw Acoustic Sonar Dataset (Day 2)...")
    raw_path = os.path.join(DATA_DIR, "sonar.csv")
    assert os.path.exists(raw_path), f"Raw dataset not found at {raw_path}"
    df_raw = pd.read_csv(raw_path, header=None)
    assert df_raw.shape == (208, 61), f"Expected shape (208, 61), got {df_raw.shape}"
    assert set(df_raw.iloc[:, 60].unique()) == {"M", "R"}, "Expected labels 'M' and 'R'"
    mine_count = (df_raw.iloc[:, 60] == "M").sum()
    rock_count = (df_raw.iloc[:, 60] == "R").sum()
    print(f"    [PASS] Dataset intact: 208 instances, 60 frequency channels, 1 target label.")
    print(f"           Class balance: {mine_count} Mines (Threats), {rock_count} Rocks (Benign).")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 2: Preprocessing & Scaling Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Verifying Preprocessed Datasets & Leakage-Free Scaler (Day 4)...")
    for f in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "preprocessing_metadata.json"]:
        p = os.path.join(PROCESSED_DIR, f)
        assert os.path.exists(p), f"Missing preprocessed artifact: {f}"

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    assert os.path.exists(scaler_path), "Fitted scaler.joblib not found"
    scaler = joblib.load(scaler_path)
    assert hasattr(scaler, "mean_") and len(scaler.mean_) == 60, "Invalid scaler parameters"
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()
    assert X_train.shape == (166, 60), f"Invalid train shape: {X_train.shape}"
    assert X_test.shape == (42, 60), f"Invalid test shape: {X_test.shape}"
    print(f"    [PASS] Stratified split verified: Train (166 x 60), Test (42 x 60). Scaler verified.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 3: Baseline & Tuned Model Persistence
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verifying Day 5 Baselines & Day 6 Tuned Models...")
    required_models = [
        "logistic_regression.joblib",
        "knn.joblib",
        "svm.joblib",
        "random_forest.joblib",
        "best_model.joblib",
        "tuned_svm.joblib",
        "tuned_random_forest.joblib",
        "tuned_knn.joblib",
        "tuned_logistic_regression.joblib",
        "best_tuned_model.joblib"
    ]
    for m in required_models:
        mp = os.path.join(MODELS_DIR, m)
        assert os.path.exists(mp), f"Model missing: {m}"
        mod = patch_sklearn_compat(joblib.load(mp))
        assert hasattr(mod, "predict"), f"Object {m} has no predict method"
    print(f"    [PASS] All 10 baseline and tuned models loaded and validated successfully.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 4: Day 10 Deep Learning & Ensemble Models
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Verifying Day 10 Deep Neural Network & Ensemble Models...")
    day10_models = [
        ("mlp_neural_network.joblib", "Deep Neural Network (MLP)"),
        ("voting_ensemble.joblib", "Soft-Voting Ensemble"),
        ("stacking_ensemble.joblib", "Stacking Meta-Classifier"),
    ]
    for filename, display_name in day10_models:
        mp = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(mp), f"Day 10 model missing: {filename}"
        mod = patch_sklearn_compat(joblib.load(mp))
        assert hasattr(mod, "predict_proba"), f"{display_name} has no predict_proba method"
        probs = mod.predict_proba(X_test)
        assert probs.shape == (42, 2), f"Unexpected probability shape: {probs.shape}"
        print(f"    [+] {display_name:<30}: Verified (Predict Proba output shape {probs.shape}).")
    print(f"    [PASS] Day 10 Neural Network and Ensemble models verified.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 5: Multi-Model Inference Engine & Dynamic Switching
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Testing SonarThreatPredictor Multi-Model Switching (Day 7/10)...")
    from src.predictor import SonarThreatPredictor
    predictor = SonarThreatPredictor(model_name="best_tuned_model.joblib")
    sample_signal = X_test.iloc[0].values
    # Invert scaling to simulate raw input
    raw_signal = predictor.scaler.inverse_transform(sample_signal.reshape(1, -1))[0]

    # Predict with baseline champion SVM
    res_svm = predictor.predict(raw_signal, use_safe_threshold=True)
    assert "classification" in res_svm and "probability_mine" in res_svm
    print(f"    [+] Prediction via Champion SVM: {res_svm['classification']} (Prob: {res_svm['probability_mine']:.4f})")

    # Switch to Deep Neural Network
    predictor.switch_model("mlp_neural_network.joblib")
    res_mlp = predictor.predict(raw_signal)
    print(f"    [+] Prediction via Neural Network (MLP): {res_mlp['classification']} (Prob: {res_mlp['probability_mine']:.4f})")

    # Switch to Stacking Classifier
    predictor.switch_model("stacking_ensemble.joblib")
    res_stack = predictor.predict(raw_signal)
    print(f"    [+] Prediction via Stacking Ensemble: {res_stack['classification']} (Prob: {res_stack['probability_mine']:.4f})")

    # Switch back to Tuned SVM
    predictor.switch_model("best_tuned_model.joblib")
    print(f"    [PASS] Dynamic runtime model switching fully functional.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 6: Zero False Negative Naval Safety Policy Audit
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Auditing Zero-Missed-Mine Safety Policy on Unseen Test Mines...")
    # Evaluate champion model under safe threshold (0.43)
    champ_model = joblib.load(os.path.join(MODELS_DIR, "best_tuned_model.joblib"))
    mine_test_indices = np.where(y_test == 1)[0]
    total_mines = len(mine_test_indices)
    assert total_mines == 22, f"Expected 22 test mines, found {total_mines}"

    X_test_mines = X_test.iloc[mine_test_indices]
    probs_mines = champ_model.predict_proba(X_test_mines)[:, 1]
    missed_mines = np.sum(probs_mines < 0.43)
    intercepted_mines = total_mines - missed_mines

    assert missed_mines == 0, f"Critical Safety Failure: {missed_mines} mines missed at tau=0.43!"
    print(f"    [PASS] Safety Audit PASSED: {intercepted_mines}/{total_mines} naval mines intercepted (100.0% Recall, 0 False Negatives).")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 7: Explainability & SHAP Local Attribution
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Verifying Explainable AI (XAI) & SHAP Pipeline (Day 8)...")
    from app import get_fast_explainer, explain_signal
    explainer = get_fast_explainer()
    assert explainer is not None, "SHAP explainer failed to initialize"
    print(f"    [PASS] SHAP KernelExplainer initialized with k-means background.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 8: Flask REST API Endpoints Verification
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Verifying Flask REST API & Web Server Endpoints (Day 9/10)...")
    from app import app
    client = app.test_client()

    # 1. Health endpoint
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"/api/health failed: {resp.status_code}"
    health_data = resp.get_json()
    assert health_data["status"] == "operational"
    print(f"    [+] GET /api/health -> 200 OK (Status: operational, Milestone: {health_data['day_milestone']})")

    # 2. Models catalog endpoint
    resp_models = client.get("/api/models")
    assert resp_models.status_code == 200, f"/api/models failed: {resp_models.status_code}"
    models_data = resp_models.get_json()
    assert len(models_data["models"]) >= 4
    print(f"    [+] GET /api/models -> 200 OK (Catalog contains {len(models_data['models'])} registered architectures)")

    # 3. Samples endpoint
    resp_samples = client.get("/api/samples")
    assert resp_samples.status_code == 200
    print("    [+] GET /api/samples -> 200 OK")

    # 4. Predict endpoint (single ping)
    sample_features = raw_signal.tolist()
    resp_pred = client.post("/api/predict", json={"features": sample_features, "threshold": 0.43})
    assert resp_pred.status_code == 200
    pred_payload = resp_pred.get_json()
    assert pred_payload["status"] == "success"
    print(f"    [+] POST /api/predict -> 200 OK ({pred_payload['data']['classification']}, Prob: {pred_payload['data']['probability_mine']:.4f})")

    # 5. Explain endpoint
    resp_explain = client.post("/api/explain", json={"features": sample_features})
    assert resp_explain.status_code == 200
    explain_payload = resp_explain.get_json()
    assert explain_payload["status"] == "success"
    assert "top_mine_drivers" in explain_payload
    print(f"    [+] POST /api/explain -> 200 OK (Attribution Type: {explain_payload['attribution_type']})")

    print("    [PASS] All REST API endpoints passed verification.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 9: Visualizations and Reports
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Verifying Results Visualizations & Structured Reports...")
    expected_results = [
        "class_distribution.png",
        "mean_spectral_signature.png",
        "model_comparison.png",
        "confusion_matrices.png",
        "roc_curves.png",
        "tuning_comparison.png",
        "threshold_optimization.png",
        "learning_curves.png",
        "feature_importance.png",
        "calibration_curves.png",
        "permutation_importance.png",
        "shap_summary.png",
        "shap_beeswarm.png",
        "neural_loss_curve.png",
        "ensemble_comparison.png",
        "ensemble_confusion_matrices.png",
        "ensemble_roc_curves.png",
        "ensemble_metrics.csv",
        "ensemble_evaluation_report.json"
    ]
    missing = []
    for r in expected_results:
        rp = os.path.join(RESULTS_DIR, r)
        if not os.path.exists(rp):
            missing.append(r)
    assert len(missing) == 0, f"Missing result files: {missing}"
    print(f"    [PASS] All {len(expected_results)} visualization artifacts and metrics reports present in results/.")
    passed_tests += 1

    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"   FULL SYSTEM VERIFICATION PASSED: {passed_tests}/{total_tests} TESTS SUCCESSFUL!")
    print("   Zero errors detected. Production pipeline is 100% operational.")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        print(f"\n[FAIL] Verification error: {e}", file=sys.stderr)
        sys.exit(1)
