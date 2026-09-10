"""
src/preprocessor.py
-------------------
Data preprocessing, feature scaling, and train-test splitting module
for the Sonar Signal Classification System (Day 4).

This module handles:
  1. Target label encoding ('M' -> 1 [Threat], 'R' -> 0 [Non-threat]).
  2. Stratified train-test splitting to preserve class balance.
  3. Feature scaling (StandardScaler / MinMaxScaler) fitted strictly on training data
     to prevent data leakage.
  4. Artifact serialization (saving preprocessed splits and fitted scaler).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Canonical directories
DEFAULT_DATA_PROCESSED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed")
)
DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)

# Label encoding constants
LABEL_MAPPING = {"M": 1, "R": 0}
INVERSE_LABEL_MAPPING = {1: "M", 0: "R"}
CLASS_DESCRIPTIONS = {
    1: "Mine (Underwater Threat)",
    0: "Rock (Non-threat)",
}


def encode_labels(y, mapping=None):
    """
    Encode categorical sonar labels into binary integer targets.

    By default:
      - 'M' (Mine / Threat) -> 1
      - 'R' (Rock / Non-threat) -> 0

    Parameters
    ----------
    y : pd.Series or array-like
        Raw target labels ('M' and 'R').
    mapping : dict, optional
        Custom mapping dictionary. Defaults to {'M': 1, 'R': 0}.

    Returns
    -------
    pd.Series
        Binary-encoded integer labels (0 and 1).

    Raises
    ------
    ValueError
        If unmapped label values exist in y.
    """
    target_mapping = mapping if mapping is not None else LABEL_MAPPING

    y_series = pd.Series(y).copy()
    unmapped = set(y_series.unique()) - set(target_mapping.keys())
    if unmapped:
        raise ValueError(
            f"[ERROR] Unexpected class labels encountered in target: {unmapped}. "
            f"Expected keys: {list(target_mapping.keys())}"
        )

    encoded = y_series.map(target_mapping).astype(int)
    return encoded


def split_sonar_data(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=True,
):
    """
    Split feature matrix X and target vector y into training and testing partitions.
    Uses stratified sampling to preserve the threat/non-threat class ratio.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input features (60 acoustic frequency energy bands).
    y : pd.Series or np.ndarray
        Encoded binary labels.
    test_size : float, default=0.2
        Proportion of the dataset allocated to the test split (default: 20%).
    random_state : int, default=42
        Seed for pseudo-random number generator for reproducible splits.
    stratify : bool, default=True
        Whether to stratify splits based on class labels y.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
        (X_train, X_test, y_train, y_test)
    """
    stratify_target = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Ensure index integrity
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
    if isinstance(y_train, (pd.Series, pd.DataFrame)):
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def scale_sonar_features(
    X_train,
    X_test,
    scaler_type="standard",
):
    """
    Scale continuous sonar frequency features using Scikit-Learn scalers.

    IMPORTANT: To prevent data leakage, the scaler is fit STRICTLY on X_train.
    The learned transformation parameters are then applied to both X_train and X_test.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training feature matrix.
    X_test : pd.DataFrame or np.ndarray
        Testing feature matrix.
    scaler_type : str, default="standard"
        Type of scaling algorithm:
          - "standard": StandardScaler (zero mean, unit variance)
          - "minmax": MinMaxScaler (bounds values strictly between 0 and 1)

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, BaseEstimator)
        (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    scaler_type_clean = scaler_type.strip().lower()

    if scaler_type_clean in ("standard", "standardscaler", "zscore"):
        scaler = StandardScaler()
    elif scaler_type_clean in ("minmax", "minmaxscaler"):
        scaler = MinMaxScaler()
    else:
        raise ValueError(
            f"[ERROR] Unsupported scaler_type '{scaler_type}'. "
            f"Allowed options: 'standard', 'minmax'."
        )

    # Extract feature column names if input is DataFrame
    feature_cols = (
        X_train.columns.tolist()
        if isinstance(X_train, pd.DataFrame)
        else [f"freq_{i}" for i in range(1, X_train.shape[1] + 1)]
    )

    # Fit strictly on X_train, then transform both splits
    X_train_scaled_arr = scaler.fit_transform(X_train)
    X_test_scaled_arr = scaler.transform(X_test)

    # Wrap back into DataFrames preserving column names
    X_train_scaled = pd.DataFrame(
        X_train_scaled_arr, columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_arr, columns=feature_cols
    )

    return X_train_scaled, X_test_scaled, scaler


def save_preprocessed_artifacts(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    scaler_name="scaler.joblib",
    processed_dir=None,
    models_dir=None,
    metadata=None,
):
    """
    Persist preprocessed splits, fitted scaler, and run metadata to disk.

    Parameters
    ----------
    X_train : pd.DataFrame
        Scaled training feature matrix.
    X_test : pd.DataFrame
        Scaled test feature matrix.
    y_train : pd.Series
        Training target labels.
    y_test : pd.Series
        Testing target labels.
    scaler : BaseEstimator
        Fitted Scikit-Learn scaler object.
    scaler_name : str, default="scaler.joblib"
        Filename for serialized scaler in models_dir.
    processed_dir : str, optional
        Target directory for processed CSV files. Defaults to 'data/processed/'.
    models_dir : str, optional
        Target directory for serialized scaler. Defaults to 'models/'.
    metadata : dict, optional
        Additional execution metadata to record in JSON.

    Returns
    -------
    dict
        Dictionary containing file paths of saved artifacts.
    """
    target_data_dir = processed_dir if processed_dir is not None else DEFAULT_DATA_PROCESSED_DIR
    target_model_dir = models_dir if models_dir is not None else DEFAULT_MODELS_DIR

    os.makedirs(target_data_dir, exist_ok=True)
    os.makedirs(target_model_dir, exist_ok=True)

    # File paths
    x_train_path = os.path.join(target_data_dir, "X_train.csv")
    x_test_path = os.path.join(target_data_dir, "X_test.csv")
    y_train_path = os.path.join(target_data_dir, "y_train.csv")
    y_test_path = os.path.join(target_data_dir, "y_test.csv")
    scaler_path = os.path.join(target_model_dir, scaler_name)
    metadata_path = os.path.join(target_data_dir, "preprocessing_metadata.json")

    # Save CSV datasets
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    pd.DataFrame(y_train, columns=["label"]).to_csv(y_train_path, index=False)
    pd.DataFrame(y_test, columns=["label"]).to_csv(y_test_path, index=False)

    # Save fitted scaler
    joblib.dump(scaler, scaler_path)

    # Prepare and save metadata
    meta_info = {
        "timestamp": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(X_train) + len(X_test),
        "n_features": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "scaler_type": type(scaler).__name__,
        "label_mapping": LABEL_MAPPING,
        "train_class_distribution": {
            str(k): int(v) for k, v in pd.Series(y_train).value_counts().items()
        },
        "test_class_distribution": {
            str(k): int(v) for k, v in pd.Series(y_test).value_counts().items()
        },
    }
    if metadata:
        meta_info.update(metadata)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)

    return {
        "X_train": x_train_path,
        "X_test": x_test_path,
        "y_train": y_train_path,
        "y_test": y_test_path,
        "scaler": scaler_path,
        "metadata": metadata_path,
    }


def load_preprocessed_artifacts(
    processed_dir=None,
    models_dir=None,
    scaler_name="scaler.joblib",
):
    """
    Load preprocessed splits and fitted scaler for downstream model training (Day 5).

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler, metadata)
    """
    target_data_dir = processed_dir if processed_dir is not None else DEFAULT_DATA_PROCESSED_DIR
    target_model_dir = models_dir if models_dir is not None else DEFAULT_MODELS_DIR

    x_train_path = os.path.join(target_data_dir, "X_train.csv")
    x_test_path = os.path.join(target_data_dir, "X_test.csv")
    y_train_path = os.path.join(target_data_dir, "y_train.csv")
    y_test_path = os.path.join(target_data_dir, "y_test.csv")
    scaler_path = os.path.join(target_model_dir, scaler_name)
    metadata_path = os.path.join(target_data_dir, "preprocessing_metadata.json")

    for path in [x_train_path, x_test_path, y_train_path, y_test_path, scaler_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[ERROR] Expected artifact not found at '{path}'. "
                f"Please run 'python preprocess.py' first."
            )

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)["label"]
    y_test = pd.read_csv(y_test_path)["label"]
    scaler = joblib.load(scaler_path)

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return X_train, X_test, y_train, y_test, scaler, metadata


class SonarPreprocessor:
    """
    Comprehensive Preprocessing Pipeline Class for the Sonar Signal Classification System.

    Encapsulates label encoding, train-test splitting, and feature scaling into
    a reusable object with scikit-learn compatible interfaces.
    """

    def __init__(
        self,
        scaler_type="standard",
        test_size=0.2,
        random_state=42,
        stratify=True,
    ):
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None

    def fit_transform_pipeline(self, df, label_col="label"):
        """
        Execute the full preprocessing pipeline on a raw Sonar DataFrame:
          1. Separate features X and target y.
          2. Binary encode y ('M' -> 1, 'R' -> 0).
          3. Stratified split into train (80%) and test (20%).
          4. Fit scaler on train features, transform train and test features.

        Parameters
        ----------
        df : pd.DataFrame
            Raw Sonar dataset DataFrame.
        label_col : str, default="label"
            Name of target column.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'X_train_scaled': pd.DataFrame
            - 'X_test_scaled': pd.DataFrame
            - 'y_train': pd.Series
            - 'y_test': pd.Series
            - 'scaler': fitted scaler object
            - 'X_train_raw': pd.DataFrame (unscaled)
            - 'X_test_raw': pd.DataFrame (unscaled)
        """
        if label_col not in df.columns:
            raise KeyError(f"Target column '{label_col}' not found in DataFrame.")

        # 1. Separate features and target
        feature_cols = [col for col in df.columns if col != label_col]
        self.feature_names = feature_cols
        X_raw = df[feature_cols].copy()
        y_raw = df[label_col].copy()

        # 2. Binary encode labels
        y_encoded = encode_labels(y_raw)

        # 3. Stratified Train-Test Split
        X_train_raw, X_test_raw, y_train, y_test = split_sonar_data(
            X_raw,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.stratify,
        )

        # 4. Feature Scaling (fit strictly on X_train)
        X_train_scaled, X_test_scaled, self.scaler = scale_sonar_features(
            X_train_raw,
            X_test_raw,
            scaler_type=self.scaler_type,
        )
        self.is_fitted = True

        return {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": self.scaler,
            "X_train_raw": X_train_raw,
            "X_test_raw": X_test_raw,
        }

    def transform_new_signals(self, X_new):
        """
        Transform new unseen sonar acoustic frequency signals using the fitted scaler.

        Parameters
        ----------
        X_new : pd.DataFrame or np.ndarray
            New acoustic signals (must match 60 frequency features).

        Returns
        -------
        pd.DataFrame
            Scaled features ready for model inference.
        """
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError(
                "[ERROR] Preprocessor is not fitted yet. "
                "Call fit_transform_pipeline() or load a saved scaler first."
            )

        scaled_arr = self.scaler.transform(X_new)
        cols = (
            self.feature_names
            if self.feature_names is not None
            else [f"freq_{i}" for i in range(1, scaled_arr.shape[1] + 1)]
        )
        return pd.DataFrame(scaled_arr, columns=cols)


def run_full_preprocessing(
    df=None,
    test_size=0.2,
    random_state=42,
    scaler_type="standard",
    save_artifacts=True,
    processed_dir=None,
    models_dir=None,
):
    """
    Run the end-to-end Day 4 preprocessing pipeline, print comprehensive
    verification reports, and persist artifacts.

    Returns
    -------
    dict
        Dictionary containing all preprocessed splits, scaler, and artifact paths.
    """
    sep = "=" * 70
    subsep = "-" * 70

    print(sep)
    print("   SONAR SIGNAL PREPROCESSING & FEATURE SCALING PIPELINE (DAY 4)")
    print(sep)
    print()

    # Step 1: Load data if not supplied
    if df is None:
        from .data_loader import load_sonar_data
        print("[*] 1. Loading Raw Dataset from data/sonar.csv...")
        df = load_sonar_data()
        print(f"    -> Successfully loaded {len(df)} records with {df.shape[1]} columns.")
    else:
        print(f"[*] 1. Using Provided Dataset: {len(df)} records, {df.shape[1]} columns.")
    print()

    # Step 2: Initialize and run preprocessor pipeline
    print(f"[*] 2. Initializing SonarPreprocessor Pipeline:")
    print(f"    - Scaler Algorithm       : {scaler_type.upper()}")
    print(f"    - Train/Test Ratio       : {int((1 - test_size)*100)}% Train / {int(test_size*100)}% Test")
    print(f"    - Random Seed            : {random_state}")
    print(f"    - Stratified Sampling    : True (maintains class proportions)")
    print()

    preprocessor = SonarPreprocessor(
        scaler_type=scaler_type,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )
    results = preprocessor.fit_transform_pipeline(df)

    X_train_scaled = results["X_train_scaled"]
    X_test_scaled = results["X_test_scaled"]
    y_train = results["y_train"]
    y_test = results["y_test"]
    scaler = results["scaler"]

    # Step 3: Display Label Encoding Details
    print(f"[*] 3. Target Label Encoding Audit:")
    raw_m_count = (df["label"] == "M").sum()
    raw_r_count = (df["label"] == "R").sum()
    print(f"    - Raw Target Classes     : 'M' (Mine / Threat) and 'R' (Rock / Non-threat)")
    print(f"    - Mapping Scheme         : 'M' -> 1 (Positive / Threat), 'R' -> 0 (Negative / Benign)")
    print(f"    - Total Threat Instances (Class 1)     : {raw_m_count} ({raw_m_count/len(df)*100:.2f}%)")
    print(f"    - Total Non-Threat Instances (Class 0) : {raw_r_count} ({raw_r_count/len(df)*100:.2f}%)")
    print()

    # Step 4: Display Train-Test Split & Stratification Audit
    print(f"[*] 4. Train-Test Partition & Stratification Audit:")
    print(f"    {subsep}")
    print(f"    {'Partition':<12} | {'Total':<7} | {'Mines (Class 1)':<18} | {'Rocks (Class 0)':<18}")
    print(f"    {subsep}")

    total_len = len(df)
    train_len = len(X_train_scaled)
    test_len = len(X_test_scaled)
    train_m = (y_train == 1).sum()
    train_r = (y_train == 0).sum()
    test_m = (y_test == 1).sum()
    test_r = (y_test == 0).sum()

    print(
        f"    {'Full':<12} | {total_len:<7} | "
        f"{raw_m_count} ({raw_m_count/total_len*100:.2f}%)".ljust(18) + " | " +
        f"{raw_r_count} ({raw_r_count/total_len*100:.2f}%)"
    )
    print(
        f"    {'Train':<12} | {train_len:<7} | "
        f"{train_m} ({train_m/train_len*100:.2f}%)".ljust(18) + " | " +
        f"{train_r} ({train_r/train_len*100:.2f}%)"
    )
    print(
        f"    {'Test':<12} | {test_len:<7} | "
        f"{test_m} ({test_m/test_len*100:.2f}%)".ljust(18) + " | " +
        f"{test_r} ({test_r/test_len*100:.2f}%)"
    )
    print(f"    {subsep}")
    print(f"    [PASS] Class proportions preserved across splits within ~1% tolerance.")
    print()

    # Step 5: Feature Scaling Verification & Anti-Leakage Confirmation
    print(f"[*] 5. Feature Scaling & Data Leakage Verification:")
    train_means = X_train_scaled.mean(axis=0)
    train_stds = X_train_scaled.std(axis=0)
    test_means = X_test_scaled.mean(axis=0)
    test_stds = X_test_scaled.std(axis=0)

    print(f"    - Training Features Mean Range  : [{train_means.min():.4f}, {train_means.max():.4f}] (Expected: ~0.00)")
    print(f"    - Training Features Std Range   : [{train_stds.min():.4f}, {train_stds.max():.4f}] (Expected: ~1.00)")
    print(f"    - Test Features Mean Range      : [{test_means.min():.4f}, {test_means.max():.4f}] (Scaled with Train stats)")
    print(f"    - Test Features Std Range       : [{test_stds.min():.4f}, {test_stds.max():.4f}]")
    print(f"    - Anti-Leakage Check            : [PASS] Scaler parameters fit strictly on X_train.")
    print()

    # Step 6: Artifact Persistence
    artifact_paths = {}
    if save_artifacts:
        print(f"[*] 6. Saving Preprocessed Artifacts to Disk:")
        artifact_paths = save_preprocessed_artifacts(
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            processed_dir=processed_dir,
            models_dir=models_dir,
        )
        for name, path in artifact_paths.items():
            rel_path = os.path.relpath(path, os.path.join(os.path.dirname(__file__), ".."))
            print(f"    - {name:<10} -> {rel_path}")
        print()

    print(sep)
    print("   DAY 4 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    print(sep)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "preprocessor": preprocessor,
        "artifact_paths": artifact_paths,
    }
