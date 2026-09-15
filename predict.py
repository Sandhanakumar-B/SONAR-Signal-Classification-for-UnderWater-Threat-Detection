"""
predict.py
----------
Day 7 Execution Script: Real-Time Sonar Signal Inference & Interactive Prediction System.

Provides an interactive Command Line Interface (CLI) for predicting underwater naval threats.
Features:
  - Automated verification (--test-samples) using known Mine and Rock samples.
  - Interactive prediction mode (--interactive).
  - Batch prediction (--batch <path/to/csv>).
"""

import sys
import argparse
import pandas as pd
import numpy as np
from src.predictor import SonarThreatPredictor

def run_test_samples():
    """Runs automated inference on known test samples."""
    print("="*60)
    print("  [Day 7] Real-Time Sonar Inference Verification  ")
    print("="*60)
    
    try:
        predictor = SonarThreatPredictor()
        print(f"[OK] Successfully loaded predictor using {predictor.model_path}")
        print(f"[OK] Safe Naval Threat Threshold: tau = {predictor.safe_threshold:.2f}")
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        sys.exit(1)
        
    try:
        # Load raw dataset to test the full pipeline (scaling + inference)
        raw_df = pd.read_csv("data/sonar.csv", header=None)
        
        # Pick one known Mine ('M') and one known Rock ('R')
        mine_row = raw_df[raw_df.iloc[:, 60] == 'M'].iloc[0]
        rock_row = raw_df[raw_df.iloc[:, 60] == 'R'].iloc[0]
        
        mine_features = mine_row.iloc[:60].values
        rock_features = rock_row.iloc[:60].values
        
        print("\n--- Test Scenario 1: Known Naval Mine (Threat) ---")
        mine_res = predictor.predict(mine_features, use_safe_threshold=True)
        print(f"Classification : {mine_res['classification']}")
        print(f"Probability    : {mine_res['probability_mine']:.4f}")
        print(f"Confidence     : {mine_res['confidence_percent']:.1f}%")
        print(f"Latency        : {mine_res['latency_ms']:.2f} ms")
        
        print("\n--- Test Scenario 2: Known Geological Rock (Benign) ---")
        rock_res = predictor.predict(rock_features, use_safe_threshold=True)
        print(f"Classification : {rock_res['classification']}")
        print(f"Probability    : {rock_res['probability_mine']:.4f}")
        print(f"Confidence     : {rock_res['confidence_percent']:.1f}%")
        print(f"Latency        : {rock_res['latency_ms']:.2f} ms")
        
    except FileNotFoundError:
        print("[ERROR] Test datasets not found in data/processed/. Please run preprocess.py first.")
    
def run_interactive():
    """Interactive mode to prompt for features."""
    print("="*60)
    print("  Interactive Sonar Signal Classification  ")
    print("="*60)
    predictor = SonarThreatPredictor()
    print("Enter 60 comma-separated acoustic frequency values (range 0.0 - 1.0).")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("Sonar Signal > ")
            if user_input.strip().lower() == 'exit':
                break
            
            features = [float(x.strip()) for x in user_input.split(',')]
            if len(features) != 60:
                print(f"[ERROR] Expected 60 features, got {len(features)}. Please try again.")
                continue
                
            res = predictor.predict(features, use_safe_threshold=True)
            print(f"\n>> {res['classification']} (Conf: {res['confidence_percent']:.1f}%, Prob: {res['probability_mine']:.3f}, Latency: {res['latency_ms']:.2f}ms)\n")
        except ValueError:
            print("[ERROR] Invalid input. Please enter numerical values separated by commas.")
        except KeyboardInterrupt:
            break

def run_batch(file_path):
    """Batch processes a CSV file of sonar signals."""
    print("="*60)
    print(f"  Batch Processing: {file_path}")
    print("="*60)
    predictor = SonarThreatPredictor()
    
    try:
        df = pd.read_csv(file_path)
        print(f"[INFO] Loaded {len(df)} signals.")
        results_df, summary = predictor.predict_batch(df)
        
        print(f"Total Signals Processed : {summary['total_signals']}")
        print(f"Threats Detected        : {summary['threats_detected']}")
        print(f"Benign Rocks            : {summary['benign_detected']}")
        print(f"Avg Latency             : {summary['average_latency_ms_per_signal']:.2f} ms/signal")
        
        output_path = "results/batch_predictions.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n[OK] Detailed predictions saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Batch processing failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 7: Sonar Signal Prediction CLI")
    parser.add_argument("--test-samples", action="store_true", help="Run automated test samples")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prediction prompt")
    parser.add_argument("--batch", type=str, help="Path to CSV file for batch processing")
    
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive()
    elif args.batch:
        run_batch(args.batch)
    elif args.test_samples:
        run_test_samples()
    else:
        # Default behavior is test samples
        run_test_samples()
