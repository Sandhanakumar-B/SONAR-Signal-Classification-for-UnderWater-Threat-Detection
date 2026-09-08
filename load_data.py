"""
load_data.py
------------
Day 2 Execution Script: Sonar Signal Dataset Loading & Exploration.
Loads the dataset via Pandas and displays dimensions, head preview,
column data types, and class distribution.
"""

import sys
from src.data_loader import load_sonar_data, display_dataset_info


def main():
    try:
        # Load dataset from data/sonar.csv using Pandas
        sonar_df = load_sonar_data()

        # Display shape, first few rows, column info, and class distribution
        display_dataset_info(sonar_df)
    except Exception as err:
        print(f"[ERROR] Failed to load dataset: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
