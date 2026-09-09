"""
eda.py
------
Day 3 Execution Script: Sonar Signal Exploratory Data Analysis (EDA).
Executes the comprehensive exploratory data analysis pipeline:
  1. Dataset summary
  2. Missing-value audit
  3. Duplicate record audit
  4. Class distribution analysis
  5. Basic statistical analysis
  6. Input features (X) and target column (y) identification
  7. Visualizations using Matplotlib / Seaborn
  8. Saving plots inside results/
  9. Verifying that no ML models are trained yet
"""

import os
import sys
from src.data_loader import load_sonar_data
from src.eda import run_full_eda


def main():
    try:
        # Load dataset from data/sonar.csv via data loader
        sonar_df = load_sonar_data()

        # Run complete Day 3 Exploratory Data Analysis workflow
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
        eda_results = run_full_eda(sonar_df, output_dir=results_dir)

        print(f"\n[INFO] All Day 3 EDA results and plots successfully generated.")
    except Exception as err:
        print(f"[ERROR] Failed to execute EDA: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
