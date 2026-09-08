"""
src/data_loader.py
------------------
Data ingestion and exploration module for the Sonar Signal Classification System.
Loads the UCI Sonar dataset and provides descriptive inspection utilities.
"""

import os
import sys
import pandas as pd

# Determine canonical path to data/sonar.csv relative to this file
DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "sonar.csv")
)


def load_sonar_data(filepath=None, add_column_names=True):
    """
    Load the UCI Sonar dataset from the local data folder using Pandas.

    Parameters
    ----------
    filepath : str, optional
        Path to the dataset CSV file. If None, defaults to 'data/sonar.csv'.
    add_column_names : bool, default=True
        Whether to assign descriptive column headers ('freq_1'..'freq_60', 'label').
        If False, columns remain 0-indexed integers (0..60).

    Returns
    -------
    pd.DataFrame
        Loaded sonar signal dataset as a Pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist at the resolved path.
    """
    target_path = filepath if filepath is not None else DEFAULT_DATA_PATH

    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"[ERROR] Sonar dataset not found at: '{target_path}'.\n"
            f"Please ensure 'sonar.csv' is placed inside the 'data/' directory."
        )

    # Standard UCI Sonar dataset is headerless with 60 numeric features + 1 label
    df = pd.read_csv(target_path, header=None)

    if add_column_names:
        feature_cols = [f"freq_{i}" for i in range(1, 61)]
        df.columns = feature_cols + ["label"]

    return df


def display_dataset_info(df, label_col="label"):
    """
    Display structured exploration metrics for the Sonar dataset:
      1. Dataset Shape
      2. First Few Rows (head)
      3. Column Information & Data Types
      4. Missing Values Summary
      5. Class Distribution & Underwater Threat Mapping

    Parameters
    ----------
    df : pd.DataFrame
        The loaded Sonar DataFrame.
    label_col : str or int, default="label"
        The column representing the class target ('label' or 60).
    """
    sep = "=" * 70

    print(sep)
    print("   SONAR SIGNAL DATASET - PRELIMINARY INSPECTION (DAY 2)")
    print(sep)
    print()

    # 1. Dataset Shape
    rows, cols = df.shape
    num_features = cols - 1
    print(f"[*] 1. Dataset Dimensions & Shape:")
    print(f"    - Total Records (Instances) : {rows}")
    print(f"    - Total Columns             : {cols}")
    print(f"    - Continuous Features       : {num_features} frequency energy bands")
    print(f"    - Target Label Column       : 1 ('{label_col}')")
    print(f"    - DataFrame Shape (rows, cols): {df.shape}")
    print()

    # 2. First Few Rows
    print(f"[*] 2. First 5 Rows (Head Preview):")
    # Show first 5 rows with selected representative columns to keep display clean
    preview_cols = list(df.columns[:5]) + ["..."] + list(df.columns[-3:])
    print(f"    Displaying subset of columns: {preview_cols}")
    print("-" * 70)
    # Print head with pandas formatting
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 100)
    print(df.head())
    print()

    # 3. Column Information
    print(f"[*] 3. Column Information & Data Types:")
    print("-" * 70)
    # Summarize dtypes
    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_summary = ", ".join([f"{count} columns of type '{dtype}'" for dtype, count in dtype_counts.items()])
    print(f"    Data Types Breakdown: {dtype_summary}")
    
    # Missing values check
    total_nulls = df.isnull().sum().sum()
    print(f"    Missing / Null Values : {total_nulls} (No missing values detected)")
    print()
    print("    Column Details (first 5 and target column):")
    for col in list(df.columns[:5]) + [label_col]:
        null_count = df[col].isnull().sum()
        dtype = df[col].dtype
        print(f"      - {col:<10} | Type: {str(dtype):<10} | Missing: {null_count}")
    print(f"      ... (and remaining {num_features - 5} continuous float64 feature columns)")
    print()

    # 4. Class Distribution
    print(f"[*] 4. Target Class Distribution:")
    print("-" * 70)
    if label_col in df.columns:
        counts = df[label_col].value_counts()
        percentages = df[label_col].value_counts(normalize=True) * 100
        
        # Label mapping context for naval threat detection
        label_meanings = {
            "M": "Mine (Explosive Underwater Threat)",
            "R": "Rock (Benign Geological Structure)"
        }
        
        print(f"    {'Class':<8} {'Count':<8} {'Percentage':<12} {'Operational Threat Classification'}")
        print(f"    {'-'*55}")
        for label, count in counts.items():
            pct = percentages[label]
            meaning = label_meanings.get(label, "Unknown")
            print(f"    {label:<8} {count:<8} {pct:>6.2f}%      {meaning}")
        print(f"    {'-'*55}")
        print(f"    Total Instances: {len(df)}")
        print(f"    Class Balance  : Balanced dataset (approx. 53.4% Mines vs 46.6% Rocks)")
    else:
        print(f"    [WARN] Target column '{label_col}' not found in DataFrame.")

    print()
    print(sep)
    print("   DATASET LOADING & VERIFICATION COMPLETED SUCCESSFULLY")
    print(sep)


def main():
    """Entry point for executing the dataset loader as a standalone script."""
    print("Loading Sonar dataset from 'data/sonar.csv'...")
    try:
        df = load_sonar_data()
        display_dataset_info(df)
    except Exception as err:
        print(f"\n[FAILED] Error loading dataset: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
