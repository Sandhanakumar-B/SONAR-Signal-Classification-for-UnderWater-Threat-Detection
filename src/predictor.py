"""
src/predictor.py
----------------
Real-time inference engine and interactive prediction module for the Sonar Signal Classification System (Day 7).

Loads serialized Day 4 preprocessing scalers and Day 6 best tuned models to perform
leakage-free, cost-safe predictions on new, unseen sonar acoustic frequency signals.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Canonical directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")


class SonarThreatPredictor:
    """
    Real-time inference engine for predicting underwater threats (Mines) vs benign rocks
    using pre-trained, tuned classification models and safe decision thresholds.
    """

    def __init__(self, model_name="best_tuned_model.joblib", scaler_name="scaler.joblib"):
        """
        Initializes the predictor by loading the trained scaler, tuned model, and diagnostic thresholds.
        """
        self.model_path = os.path.join(MODELS_DIR, model_name)
        self.scaler_path = os.path.join(MODELS_DIR, scaler_name)
        self.diagnostics_path = os.path.join(RESULTS_DIR, "tuning_and_diagnostics.json")

        self.model = None
        self.scaler = None
        self.safe_threshold = 0.50
        
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, scaler, and optimized thresholds from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Did you run train.py and tune.py?")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}. Did you run preprocess.py?")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        # Attempt to load cost-safe optimized threshold from Day 6 diagnostics
        try:
            if os.path.exists(self.diagnostics_path):
                with open(self.diagnostics_path, 'r') as f:
                    diagnostics = json.load(f)
                    # Extract the safe high-recall threshold (default 0.43 from Day 6)
                    self.safe_threshold = diagnostics.get("operating_points", {}).get("high_recall_safe", {}).get("threshold", 0.50)
            else:
                print(f"[WARN] Diagnostics JSON not found at {self.diagnostics_path}. Using default 0.50 threshold.")
        except Exception as e:
            print(f"[WARN] Could not parse threshold from diagnostics: {e}. Using 0.50.")

    def preprocess_input(self, features):
        """
        Validates and scales raw input features.
        
        Parameters
        ----------
        features : list, np.ndarray, or pd.Series
            Raw 60-band acoustic frequency signal.
            
        Returns
        -------
        np.ndarray
            Shape (1, 60) standardized feature array.
        """
        features_arr = np.array(features).astype(float)
        
        # Validation
        if features_arr.shape == (60,):
            features_arr = features_arr.reshape(1, -1)
        elif features_arr.shape != (1, 60):
            raise ValueError(f"Expected 60 acoustic frequency bands, got {features_arr.shape[1] if len(features_arr.shape) > 1 else len(features_arr)}")
            
        # Standardize using fitted scaler (no data leakage)
        if hasattr(self.scaler, "feature_names_in_"):
            features_df = pd.DataFrame(features_arr, columns=self.scaler.feature_names_in_)
            scaled_features = self.scaler.transform(features_df)
        else:
            scaled_features = self.scaler.transform(features_arr)
        return scaled_features

    def predict(self, features, use_safe_threshold=True, threshold=None):
        """
        Performs a prediction on a single sonar signal.
        
        Parameters
        ----------
        features : list, np.ndarray, or pd.Series
            60 raw frequency bands.
        use_safe_threshold : bool
            If True, uses the optimized high-recall threshold to ensure 0 missed mines.
        threshold : float or None
            Custom decision threshold in [0, 1]. If provided, overrides use_safe_threshold.
            
        Returns
        -------
        dict
            Structured prediction result including probabilities, classes, and confidence.
        """
        start_time = time.perf_counter()
        
        X_scaled = self.preprocess_input(features)
        if hasattr(self.model, "feature_names_in_"):
            X_eval = pd.DataFrame(X_scaled, columns=self.model.feature_names_in_)
        else:
            X_eval = X_scaled
        
        # Get probabilities (Class 1 = Mine, Class 0 = Rock)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_eval)[0]
            prob_mine = probs[1]
        else:
            # Fallback if model doesn't support probability
            pred = self.model.predict(X_eval)[0]
            prob_mine = 1.0 if pred == 1 else 0.0

        # Apply threshold
        if threshold is not None:
            applied_threshold = float(threshold)
        else:
            applied_threshold = self.safe_threshold if use_safe_threshold else 0.50

        is_mine = prob_mine >= applied_threshold
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Confidence score (distance from threshold)
        if is_mine:
            confidence = ((prob_mine - applied_threshold) / (1.0 - applied_threshold)) * 100 if applied_threshold < 1.0 else 100.0
        else:
            confidence = ((applied_threshold - prob_mine) / applied_threshold) * 100 if applied_threshold > 0.0 else 100.0

        result = {
            "is_threat": bool(is_mine),
            "classification": "Mine (Threat Detected)" if is_mine else "Rock (Benign Object)",
            "probability_mine": float(prob_mine),
            "probability_rock": float(1.0 - prob_mine),
            "applied_threshold": float(applied_threshold),
            "confidence_percent": float(confidence),
            "latency_ms": float(latency_ms)
        }
        return result

    def predict_batch(self, df, use_safe_threshold=True, threshold=None):
        """
        Performs vectorized predictions on a batch DataFrame of signals.
        """
        start_time = time.perf_counter()
        
        if df.shape[1] != 60:
            raise ValueError(f"Batch prediction requires exactly 60 features, got {df.shape[1]}")
            
        X_scaled = self.scaler.transform(df.values)
        
        if hasattr(self.model, "predict_proba"):
            probs_mine = self.model.predict_proba(X_scaled)[:, 1]
        else:
            preds = self.model.predict(X_scaled)
            probs_mine = preds.astype(float)
            
        if threshold is not None:
            applied_threshold = float(threshold)
        else:
            applied_threshold = self.safe_threshold if use_safe_threshold else 0.50

        predictions = (probs_mine >= applied_threshold).astype(int)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        results_df = df.copy()
        results_df["Probability_Mine"] = probs_mine
        results_df["Probability_Rock"] = 1.0 - probs_mine
        results_df["Prediction"] = ["Mine" if p == 1 else "Rock" for p in predictions]
        
        summary = {
            "total_signals": len(df),
            "threats_detected": int(sum(predictions)),
            "benign_detected": int(len(df) - sum(predictions)),
            "average_latency_ms_per_signal": latency_ms / len(df)
        }
        
        return results_df, summary
