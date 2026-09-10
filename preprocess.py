"""
preprocess.py
-------------
Day 4 Execution Script: Sonar Signal Data Preprocessing, Scaling & Splitting.
Executes the comprehensive Day 4 preprocessing pipeline:
  1. Loads raw sonar acoustic frequency dataset from data/sonar.csv
  2. Binary encodes target labels ('M' -> 1 [Threat], 'R' -> 0 [Non-threat])
  3. Performs stratified train-test splitting (80% Train [166] / 20% Test [42])
  4. Fits StandardScaler strictly on training features (zero data leakage)
  5. Transforms test features with fitted scaler
  6. Saves processed datasets (data/processed/) and serialized scaler (models/scaler.joblib)
  7. Audits statistical scaling integrity and stratified class distribution
"""

import sys
from src.data_loader import load_sonar_data
from src.preprocessor import run_full_preprocessing


def main():
    try:
        # Load raw dataset via existing Day 2 data loader
        sonar_df = load_sonar_data()

        # Run complete Day 4 preprocessing pipeline
        results = run_full_preprocessing(
            df=sonar_df,
            test_size=0.2,
            random_state=42,
            scaler_type="standard",
            save_artifacts=True,
        )

        print("\n[INFO] All Day 4 preprocessing steps and artifacts successfully completed.")
    except Exception as err:
        print(f"[ERROR] Failed to execute Day 4 preprocessing: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
