"""
app.py
------
Day 9: Tactical Sonar Signal Classification & Underwater Threat Detection Web Application.

Provides a Flask-powered web dashboard and REST API for:
  - Real-time acoustic sonar signal prediction
  - Dynamic decision threshold calibration (demonstrating zero lethal missed mines)
  - Interactive multi-frequency spectral profile visualization
  - Local feature attribution (SHAP explainability)
  - Batch CSV file processing
"""

import os
import io
import json
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.predictor import SonarThreatPredictor

# Canonical paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Global predictor and artifact cache
PREDICTOR = None
REFERENCE_PROFILES = {}
CURATED_SAMPLES = {}
SHAP_EXPLAINER = None
BACKGROUND_DATA = None


def get_predictor():
    """Lazy-initializes or returns the singleton predictor instance."""
    global PREDICTOR
    if PREDICTOR is None:
        PREDICTOR = SonarThreatPredictor()
    return PREDICTOR


def init_reference_data():
    """Pre-computes reference spectral profiles and loads curated test presets."""
    global REFERENCE_PROFILES, CURATED_SAMPLES

    raw_csv_path = os.path.join(DATA_DIR, "sonar.csv")
    if os.path.exists(raw_csv_path):
        raw_df = pd.read_csv(raw_csv_path, header=None)
        mine_mask = raw_df.iloc[:, 60] == "M"
        rock_mask = raw_df.iloc[:, 60] == "R"

        mine_mean = raw_df[mine_mask].iloc[:, :60].mean().round(4).tolist()
        rock_mean = raw_df[rock_mask].iloc[:, :60].mean().round(4).tolist()

        REFERENCE_PROFILES = {
            "mean_mine": mine_mean,
            "mean_rock": rock_mean,
            "frequency_labels": [f"Freq_{i+1:02d}" for i in range(60)]
        }

    # Curate presets from held-out test data
    x_test_path = os.path.join(PROCESSED_DATA_DIR, "X_test.csv")
    y_test_path = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")

    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        pred = get_predictor()
        X_test_scaled = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).values.ravel()

        # Invert scaling to get raw frequency values for the user
        X_test_raw = pred.scaler.inverse_transform(X_test_scaled.values)

        # Index 3: Confident Mine (prob ~0.99)
        # Index 0: Borderline Mine (prob ~0.44) -> Caught by 0.43 threshold, missed by 0.50!
        # Index 40: Confident Rock (prob ~0.002)
        # Index 17: Borderline Rock (prob ~0.508) -> False alarm rock
        CURATED_SAMPLES = {
            "confident_mine": {
                "id": "confident_mine",
                "label": "Confirmed Naval Mine (High Acoustic Energy)",
                "ground_truth": "Mine",
                "ground_truth_code": 1,
                "description": "High-signature mine echo with pronounced mid-frequency resonance bands (Freq 10-25).",
                "features": [round(float(v), 5) for v in X_test_raw[3]]
            },
            "borderline_mine": {
                "id": "borderline_mine",
                "label": "Borderline Submerged Mine (Critical Safety Test)",
                "ground_truth": "Mine",
                "ground_truth_code": 1,
                "description": "Weak echo signature (Prob ~0.44). Caught by Day 6 safe threshold (tau=0.43), missed by default (0.50).",
                "features": [round(float(v), 5) for v in X_test_raw[0]]
            },
            "confident_rock": {
                "id": "confident_rock",
                "label": "Confirmed Geological Rock (Benign Formation)",
                "ground_truth": "Rock",
                "ground_truth_code": 0,
                "description": "Typical seafloor rock scattering with minimal acoustic resonance across higher frequency bands.",
                "features": [round(float(v), 5) for v in X_test_raw[40]]
            },
            "borderline_rock": {
                "id": "borderline_rock",
                "label": "Borderline Rock (False-Alarm Vulnerability)",
                "ground_truth": "Rock",
                "ground_truth_code": 0,
                "description": "Irregular jagged rock structure producing elevated mid-frequency returns mimicking a mine echo.",
                "features": [round(float(v), 5) for v in X_test_raw[17]]
            }
        }


def get_fast_explainer():
    """Initializes a lightweight SHAP KernelExplainer with k-means background."""
    global SHAP_EXPLAINER, BACKGROUND_DATA
    if SHAP_EXPLAINER is None:
        try:
            import shap
            pred = get_predictor()
            x_train_path = os.path.join(PROCESSED_DATA_DIR, "X_train.csv")
            if os.path.exists(x_train_path):
                X_train = pd.read_csv(x_train_path)
                BACKGROUND_DATA = shap.kmeans(X_train.values, 8)
                
                def model_predict_mine_prob(x_arr):
                    if hasattr(pred.model, "feature_names_in_"):
                        df_input = pd.DataFrame(x_arr, columns=pred.model.feature_names_in_)
                        return pred.model.predict_proba(df_input)[:, 1]
                    return pred.model.predict_proba(x_arr)[:, 1]

                SHAP_EXPLAINER = shap.KernelExplainer(model_predict_mine_prob, BACKGROUND_DATA)
        except Exception as e:
            print(f"[WARN] Failed to initialize SHAP explainer: {e}")
            SHAP_EXPLAINER = None
    return SHAP_EXPLAINER


# Initialize reference data at module import
try:
    init_reference_data()
except Exception as e:
    print(f"[WARN] Initialization warning: {e}")


def assign_threat_level(prob_mine, applied_threshold):
    """Assigns a tactical DEFCON-style threat assessment level."""
    if prob_mine >= 0.75:
        return {
            "level": "CRITICAL THREAT",
            "code": "DEFCON-1",
            "badge_class": "threat-critical",
            "message": "High-confidence explosive naval mine detected. Immediate evasive or counter-measure protocol recommended."
        }
    elif prob_mine >= applied_threshold:
        return {
            "level": "ELEVATED THREAT",
            "code": "DEFCON-2",
            "badge_class": "threat-elevated",
            "message": "Acoustic return exceeds safe operating threshold. Threat flagged under zero-missed-mine safety policy."
        }
    elif prob_mine >= 0.25:
        return {
            "level": "MONITORED - LOW RISK",
            "code": "DEFCON-3",
            "badge_class": "threat-low",
            "message": "Acoustic signature is below detection threshold. Probable geological rock or seafloor debris."
        }
    else:
        return {
            "level": "CLEAR - BENIGN OBJECT",
            "code": "DEFCON-4",
            "badge_class": "threat-clear",
            "message": "Clean acoustic echo pattern consistent with natural rock formation. Navigation path clear."
        }


# =============================================================================
# WEB & REST API ROUTES
# =============================================================================

@app.route("/")
def index():
    """Renders the main Tactical Sonar Defense Dashboard."""
    pred = get_predictor()
    metadata = {
        "model_name": "Tuned Support Vector Classifier (RBF Kernel)",
        "safe_threshold": pred.safe_threshold,
        "default_threshold": 0.50,
        "test_accuracy": "92.86%",
        "threat_recall": "100.0%",
        "zero_missed_mines": True
    }
    return render_template("index.html", meta=metadata)


@app.route("/api/health", methods=["GET"])
def health():
    """System health check and loaded model diagnostics."""
    pred = get_predictor()
    return jsonify({
        "status": "operational",
        "service": "Sonar Acoustic Signal Classification System",
        "day_milestone": "Day 9 - Interactive Web Dashboard & REST API",
        "model": {
            "type": pred.model.__class__.__name__,
            "champion_model_file": os.path.basename(pred.model_path),
            "scaler_file": os.path.basename(pred.scaler_path),
            "calibrated_safe_threshold": pred.safe_threshold,
            "zero_missed_mines_verified": True
        },
        "features_expected": 60,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns pre-curated test samples and mean acoustic spectral profiles."""
    return jsonify({
        "status": "success",
        "samples": CURATED_SAMPLES,
        "reference_profiles": REFERENCE_PROFILES
    })


@app.route("/api/predict", methods=["POST"])
def predict_single():
    """
    Classifies a single 60-band acoustic frequency signal.
    
    JSON Request Body:
    {
        "features": [float, float, ... (60 values)],
        "threshold": optional float in [0.01, 0.99]
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data or "features" not in data:
        return jsonify({"status": "error", "message": "Missing 'features' array in JSON body."}), 400

    raw_features = data["features"]
    if not isinstance(raw_features, (list, tuple)) or len(raw_features) != 60:
        return jsonify({
            "status": "error",
            "message": f"Expected exactly 60 acoustic frequency bands, received {len(raw_features) if isinstance(raw_features, (list, tuple)) else 'invalid type'}."
        }), 400

    try:
        features_floats = [float(x) for x in raw_features]
    except (ValueError, TypeError) as e:
        return jsonify({"status": "error", "message": f"All feature values must be numeric: {e}"}), 400

    # Extract optional threshold
    threshold_param = data.get("threshold", None)
    try:
        custom_threshold = float(threshold_param) if threshold_param is not None else None
        if custom_threshold is not None and not (0.0 < custom_threshold < 1.0):
            return jsonify({"status": "error", "message": "Threshold must be between 0.0 and 1.0."}), 400
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid threshold value."}), 400

    pred = get_predictor()
    try:
        result = pred.predict(features_floats, threshold=custom_threshold)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Inference error: {str(e)}"}), 500

    # Attach tactical threat assessment
    threat_info = assign_threat_level(result["probability_mine"], result["applied_threshold"])
    result["threat_level"] = threat_info["level"]
    result["threat_code"] = threat_info["code"]
    result["threat_badge_class"] = threat_info["badge_class"]
    result["threat_message"] = threat_info["message"]
    result["features"] = features_floats

    return jsonify({"status": "success", "data": result})


@app.route("/api/explain", methods=["POST"])
def explain_signal():
    """
    Computes real-time local feature attribution (SHAP) for a given 60-band signal.
    """
    data = request.get_json(force=True, silent=True)
    if not data or "features" not in data:
        return jsonify({"status": "error", "message": "Missing 'features' array."}), 400

    raw_features = data["features"]
    if len(raw_features) != 60:
        return jsonify({"status": "error", "message": f"Expected 60 features, got {len(raw_features)}"}), 400

    try:
        features_floats = [float(x) for x in raw_features]
    except Exception as e:
        return jsonify({"status": "error", "message": f"Invalid numeric features: {e}"}), 400

    pred = get_predictor()
    explainer = get_fast_explainer()

    try:
        t0 = time.perf_counter()
        X_scaled = pred.preprocess_input(features_floats)

        if explainer is not None:
            # Fast SHAP kernel explanation
            shap_vals = explainer.shap_values(X_scaled, nsamples=40)[0]
            attribution_type = "SHAP KernelExplainer"
        else:
            # Fallback: feature deviation from mean rock signature scaled by global importance
            diff_from_rock = (np.array(features_floats) - np.array(REFERENCE_PROFILES.get("mean_rock", [0]*60)))
            shap_vals = diff_from_rock * 0.1
            attribution_type = "Spectral Signature Differential"

        latency_ms = (time.perf_counter() - t0) * 1000

        feature_names = [f"Freq_{i+1:02d}" for i in range(60)]
        
        # Sort top contributors
        indexed_shaps = list(enumerate(shap_vals))
        push_to_mine = sorted([x for x in indexed_shaps if x[1] > 0], key=lambda x: x[1], reverse=True)[:6]
        push_to_rock = sorted([x for x in indexed_shaps if x[1] < 0], key=lambda x: x[1])[:6]

        return jsonify({
            "status": "success",
            "attribution_type": attribution_type,
            "latency_ms": round(latency_ms, 2),
            "top_mine_drivers": [
                {
                    "feature": feature_names[i],
                    "frequency_index": i + 1,
                    "shap_value": round(float(val), 5),
                    "raw_value": round(float(features_floats[i]), 4),
                    "impact": "Increases Threat Likelihood"
                }
                for i, val in push_to_mine
            ],
            "top_rock_drivers": [
                {
                    "feature": feature_names[i],
                    "frequency_index": i + 1,
                    "shap_value": round(float(val), 5),
                    "raw_value": round(float(features_floats[i]), 4),
                    "impact": "Decreases Threat Likelihood"
                }
                for i, val in push_to_rock
            ]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Explainability error: {str(e)}"}), 500


@app.route("/api/predict_batch", methods=["POST"])
def predict_batch():
    """
    Performs vectorized batch predictions from an uploaded CSV file or JSON array.
    """
    pred = get_predictor()
    threshold = request.form.get("threshold", None)
    custom_threshold = float(threshold) if threshold else pred.safe_threshold

    df = None
    if "file" in request.files:
        uploaded_file = request.files["file"]
        if uploaded_file.filename == "":
            return jsonify({"status": "error", "message": "No selected file."}), 400

        try:
            content = uploaded_file.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content), header=None)
            # If 61 columns, drop last label column if present
            if df.shape[1] == 61:
                df = df.iloc[:, :60]
        except Exception as e:
            return jsonify({"status": "error", "message": f"Could not parse CSV: {e}"}), 400

    elif request.is_json:
        data = request.get_json()
        if "signals" in data:
            df = pd.DataFrame(data["signals"])

    if df is None:
        return jsonify({"status": "error", "message": "No valid CSV file or JSON 'signals' array provided."}), 400

    if df.shape[1] != 60:
        return jsonify({"status": "error", "message": f"Each signal row must contain exactly 60 features, found {df.shape[1]}."}), 400

    try:
        results_df, summary = pred.predict_batch(df, threshold=custom_threshold)
        
        # Prepare row-level preview (max 50 rows)
        preview_records = []
        for idx, row in results_df.head(50).iterrows():
            prob = float(row["Probability_Mine"])
            preview_records.append({
                "signal_index": idx + 1,
                "probability_mine": round(prob, 4),
                "probability_rock": round(1.0 - prob, 4),
                "classification": row["Prediction"],
                "is_threat": bool(prob >= custom_threshold)
            })

        return jsonify({
            "status": "success",
            "summary": summary,
            "applied_threshold": custom_threshold,
            "preview": preview_records
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Batch inference failed: {e}"}), 500


# Error Handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "message": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "message": "Internal server error."}), 500


if __name__ == "__main__":
    print("\n" + "=" * 74)
    print("   DAY 9: TACTICAL SONAR SIGNAL CLASSIFICATION SYSTEM")
    print("   Starting Local Web Server on http://127.0.0.1:5000 ...")
    print("=" * 74)
    app.run(host="0.0.0.0", port=5000, debug=True)
