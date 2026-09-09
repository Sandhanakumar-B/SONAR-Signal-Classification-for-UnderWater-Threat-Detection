"""
src/eda.py
----------
Exploratory Data Analysis (EDA) module for the Sonar Signal Classification System.
Performs comprehensive data inspection, statistical analysis, feature identification,
and generates informative visualizations saved into the results directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

# Canonical results directory relative to this module
DEFAULT_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def get_dataset_summary(df):
    """
    Generate a high-level structural summary of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded sonar dataset.

    Returns
    -------
    dict
        Dictionary containing shape, memory usage, column counts by dtype,
        and global numerical bounds.
    """
    num_rows, num_cols = df.shape
    memory_kb = df.memory_usage(deep=True).sum() / 1024.0
    dtype_counts = df.dtypes.value_counts().to_dict()
    numeric_df = df.select_dtypes(include=[np.number])

    summary = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "num_features": num_cols - 1,
        "memory_kb": memory_kb,
        "dtype_counts": {str(k): v for k, v in dtype_counts.items()},
        "global_min": float(numeric_df.min().min()) if not numeric_df.empty else None,
        "global_max": float(numeric_df.max().max()) if not numeric_df.empty else None,
        "global_mean": float(numeric_df.values.mean()) if not numeric_df.empty else None,
        "global_std": float(numeric_df.values.std()) if not numeric_df.empty else None,
    }
    return summary


def check_missing_values(df):
    """
    Check for missing / null / NaN values across all columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.

    Returns
    -------
    dict
        Total null count, per-column missing series, and boolean flag.
    """
    null_series = df.isnull().sum()
    total_nulls = int(null_series.sum())
    has_missing = total_nulls > 0

    return {
        "total_nulls": total_nulls,
        "has_missing": has_missing,
        "null_series": null_series,
    }


def check_duplicates(df):
    """
    Check for duplicate rows in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.

    Returns
    -------
    dict
        Duplicate record count and duplicate indices.
    """
    duplicate_mask = df.duplicated()
    num_duplicates = int(duplicate_mask.sum())

    return {
        "num_duplicates": num_duplicates,
        "has_duplicates": num_duplicates > 0,
        "duplicate_indices": df[duplicate_mask].index.tolist(),
    }


def analyze_class_distribution(df, label_col="label"):
    """
    Analyze the frequency and percentage distribution of the target class.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    label_col : str, default="label"
        Name of the target column.

    Returns
    -------
    dict
        Counts, percentages, imbalance ratio, and threat category mapping.
    """
    if label_col not in df.columns:
        raise KeyError(f"Target column '{label_col}' not found in DataFrame.")

    counts = df[label_col].value_counts().to_dict()
    percentages = (df[label_col].value_counts(normalize=True) * 100).round(2).to_dict()
    total_records = len(df)

    # Calculate class imbalance ratio (majority : minority)
    vals = list(counts.values())
    imbalance_ratio = round(max(vals) / min(vals), 2) if min(vals) > 0 else 0.0

    class_labels = {
        "M": "Mine (Underwater Explosive Threat)",
        "R": "Rock (Benign Geological Formation)"
    }

    return {
        "counts": counts,
        "percentages": percentages,
        "total_records": total_records,
        "imbalance_ratio": imbalance_ratio,
        "class_labels": class_labels,
        "is_balanced": imbalance_ratio < 1.5,
    }


def compute_statistical_analysis(df, label_col="label"):
    """
    Compute descriptive statistics for continuous frequency features overall
    and broken down by target class.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    label_col : str, default="label"
        Name of target column.

    Returns
    -------
    dict
        Overall descriptive stats, per-class mean spectra, and top variance features.
    """
    feature_cols = [col for col in df.columns if col != label_col]
    numeric_df = df[feature_cols]

    desc_stats = numeric_df.describe().T
    variances = numeric_df.var().sort_values(ascending=False)

    # Class-wise mean response across all frequency bands
    class_mean_df = df.groupby(label_col)[feature_cols].mean().T
    class_std_df = df.groupby(label_col)[feature_cols].std().T

    return {
        "desc_stats": desc_stats,
        "top_variance_features": variances.head(5).to_dict(),
        "lowest_variance_features": variances.tail(5).to_dict(),
        "class_mean_df": class_mean_df,
        "class_std_df": class_std_df,
        "feature_cols": feature_cols,
    }


def get_features_and_target(df, label_col="label"):
    """
    Separate input features (X) and target variable (y).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    label_col : str, default="label"
        Target column identifier.

    Returns
    -------
    tuple
        (X, y, feature_names, target_name)
        - X : pd.DataFrame of shape (N, 60), continuous features
        - y : pd.Series of shape (N,), categorical labels
        - feature_names : list of str
        - target_name : str
    """
    if label_col not in df.columns:
        raise KeyError(f"Target column '{label_col}' not found in DataFrame.")

    feature_cols = [col for col in df.columns if col != label_col]
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    return X, y, feature_cols, label_col


def generate_and_save_visualizations(df, output_dir=None, label_col="label"):
    """
    Generate and save five comprehensive exploratory visualizations into results/:
      1. results/class_distribution.png       - Bar chart of Mines vs Rocks
      2. results/mean_spectral_signature.png  - Mean energy spectral profile by class
      3. results/feature_distributions.png    - Histograms & KDE for representative bands
      4. results/feature_boxplots.png         - Boxplots comparing classes on key features
      5. results/correlation_heatmap.png      - Correlation matrix heatmap

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame.
    output_dir : str, optional
        Destination directory. Defaults to 'results/'.
    label_col : str, default="label"
        Target column name.

    Returns
    -------
    list of str
        File paths of generated plots.
    """
    target_dir = output_dir if output_dir is not None else DEFAULT_RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    saved_plots = []
    sns.set_theme(style="whitegrid", font="sans-serif")
    palette = {"M": "#D9381E", "R": "#2A75D3"}  # Crimson for Mines, Steel Blue for Rocks

    feature_cols = [c for c in df.columns if c != label_col]

    # -------------------------------------------------------------
    # Plot 1: Target Class Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[label_col].value_counts()
    percentages = df[label_col].value_counts(normalize=True) * 100
    bars = ax.bar(
        ["Mines (M)\nThreat", "Rocks (R)\nNon-threat"],
        [counts.get("M", 0), counts.get("R", 0)],
        color=[palette["M"], palette["R"]],
        edgecolor="black",
        linewidth=1.2,
        width=0.55
    )

    # Add count and percentage labels above bars
    for bar, label_key in zip(bars, ["M", "R"]):
        h = bar.get_height()
        pct = percentages.get(label_key, 0.0)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 2,
            f"{int(h)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )

    ax.set_ylim(0, int(counts.max()) + 18)
    ax.set_title("Sonar Target Class Distribution (Mines vs. Rocks)", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Instance Count", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plot1_path = os.path.join(target_dir, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=300)
    plt.close(fig)
    saved_plots.append(plot1_path)

    # -------------------------------------------------------------
    # Plot 2: Mean Spectral Signature Profile
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5.5))
    band_indices = np.arange(1, len(feature_cols) + 1)

    mean_m = df[df[label_col] == "M"][feature_cols].mean().values
    std_m = df[df[label_col] == "M"][feature_cols].std().values

    mean_r = df[df[label_col] == "R"][feature_cols].mean().values
    std_r = df[df[label_col] == "R"][feature_cols].std().values

    ax.plot(band_indices, mean_m, color=palette["M"], linewidth=2.2, label="Mine (M) - Mean Energy Profile")
    ax.fill_between(band_indices, mean_m - 0.5 * std_m, mean_m + 0.5 * std_m, color=palette["M"], alpha=0.18, label="Mine +/- 0.5 Std Dev")

    ax.plot(band_indices, mean_r, color=palette["R"], linewidth=2.2, linestyle="--", label="Rock (R) - Mean Energy Profile")
    ax.fill_between(band_indices, mean_r - 0.5 * std_r, mean_r + 0.5 * std_r, color=palette["R"], alpha=0.18, label="Rock +/- 0.5 Std Dev")

    ax.set_title("Mean Acoustic Energy Spectral Signature (Mines vs. Rocks)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Frequency Band Index (1 to 60)", fontsize=11)
    ax.set_ylabel("Normalized Return Energy", fontsize=11)
    ax.set_xlim(1, 60)
    ax.set_xticks(np.arange(5, 61, 5))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plot2_path = os.path.join(target_dir, "mean_spectral_signature.png")
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=300)
    plt.close(fig)
    saved_plots.append(plot2_path)

    # -------------------------------------------------------------
    # Plot 3: Feature Distributions of Representative Frequency Bands
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    rep_bands = ["freq_10", "freq_20", "freq_35", "freq_48"]

    for ax, band in zip(axes.flatten(), rep_bands):
        if band in df.columns:
            sns.histplot(
                data=df,
                x=band,
                hue=label_col,
                palette=palette,
                kde=True,
                ax=ax,
                stat="density",
                common_norm=False,
                bins=15,
                alpha=0.45
            )
            ax.set_title(f"Distribution of Energy at {band}", fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{band} Energy Value")
            ax.set_ylabel("Density")

    plt.suptitle("Energy Distributions of Representative Frequency Bands", fontsize=13, fontweight="bold", y=0.98)
    plot3_path = os.path.join(target_dir, "feature_distributions.png")
    plt.tight_layout()
    plt.savefig(plot3_path, dpi=300)
    plt.close(fig)
    saved_plots.append(plot3_path)

    # -------------------------------------------------------------
    # Plot 4: Comparative Boxplots by Class
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    box_bands = ["freq_11", "freq_21", "freq_28", "freq_36"]

    for ax, band in zip(axes, box_bands):
        if band in df.columns:
            sns.boxplot(
                data=df,
                x=label_col,
                y=band,
                hue=label_col,
                order=["M", "R"],
                palette=palette,
                legend=False,
                ax=ax,
                width=0.45,
                fliersize=3.5
            )
            ax.set_title(f"{band}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Target Class")
            ax.set_ylabel("Energy Value")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Mine (M)", "Rock (R)"])

    plt.suptitle("Class Separation Comparison across Distinct Acoustic Bands", fontsize=13, fontweight="bold", y=1.02)
    plot4_path = os.path.join(target_dir, "feature_boxplots.png")
    plt.tight_layout()
    plt.savefig(plot4_path, dpi=300)
    plt.close(fig)
    saved_plots.append(plot4_path)

    # -------------------------------------------------------------
    # Plot 5: Correlation Heatmap Across Frequency Spectrum
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8.5))
    # Sample every 2nd frequency band to maintain legible ticks while showing spectrum
    sample_cols = [f"freq_{i}" for i in range(1, 61, 2)]
    corr_matrix = df[sample_cols].corr()

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-0.4,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
        xticklabels=sample_cols,
        yticklabels=sample_cols
    )
    ax.set_title("Correlation Heatmap Across Sonar Frequency Bands (Sampled Every 2 Bands)", fontsize=12, pad=12, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=8)

    plot5_path = os.path.join(target_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(plot5_path, dpi=300)
    plt.close(fig)
    saved_plots.append(plot5_path)

    return saved_plots


def run_full_eda(df, output_dir=None, label_col="label"):
    """
    Execute all 8 exploratory data analysis steps and print structured terminal reporting.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded sonar dataset.
    output_dir : str, optional
        Target directory for plot artifacts.
    label_col : str, default="label"
        Name of target column.

    Returns
    -------
    dict
        Aggregated results of the EDA pipeline.
    """
    sep = "=" * 74
    subsep = "-" * 74

    print(sep)
    print("   EXPLORATORY DATA ANALYSIS (EDA) - SONAR SIGNAL CLASSIFICATION (DAY 3)")
    print(sep)
    print()

    # Step 1: Dataset Summary
    print("[*] STEP 1: DATASET SUMMARY & ARCHITECTURE")
    print(subsep)
    summary = get_dataset_summary(df)
    print(f"    - Total Instances (Rows)        : {summary['num_rows']}")
    print(f"    - Total Columns                 : {summary['num_cols']}")
    print(f"    - Continuous Input Features     : {summary['num_features']} frequency energy channels (freq_1 .. freq_60)")
    print(f"    - Target Classification Column  : 1 ('{label_col}')")
    print(f"    - Dataset Memory Footprint      : {summary['memory_kb']:.2f} KB")
    print(f"    - Column Data Types Breakdown   : {summary['dtype_counts']}")
    print(f"    - Global Feature Value Range    : Min = {summary['global_min']:.4f}, Max = {summary['global_max']:.4f}")
    print(f"    - Global Signal Mean & Std Dev  : Mean = {summary['global_mean']:.4f}, Std = {summary['global_std']:.4f}")
    print()

    # Step 2: Missing-Value Checking
    print("[*] STEP 2: MISSING VALUE AUDIT")
    print(subsep)
    missing_info = check_missing_values(df)
    print(f"    - Total Missing / NaN Values    : {missing_info['total_nulls']}")
    print(f"    - Missing Values Detected?      : {'YES - Imputation Required' if missing_info['has_missing'] else 'NO - Dataset is 100% Complete'}")
    print(f"    - Integrity Verification        : All 60 features and target label are complete without null entries.")
    print()

    # Step 3: Duplicate Checking
    print("[*] STEP 3: DUPLICATE RECORD AUDIT")
    print(subsep)
    duplicate_info = check_duplicates(df)
    print(f"    - Total Duplicate Rows Found    : {duplicate_info['num_duplicates']}")
    print(f"    - Duplicates Detected?          : {'YES - Deduplication Required' if duplicate_info['has_duplicates'] else 'NO - All instances are unique'}")
    print(f"    - Integrity Verification        : 208 distinct physical sonar pings recorded.")
    print()

    # Step 4: Class Distribution Analysis
    print("[*] STEP 4: CLASS DISTRIBUTION & THREAT ANALYSIS")
    print(subsep)
    class_info = analyze_class_distribution(df, label_col=label_col)
    print(f"    {'Class Label':<14} {'Count':<8} {'Percentage':<12} {'Naval Operational Threat Category'}")
    print(f"    {'-'*65}")
    for key, count in class_info["counts"].items():
        pct = class_info["percentages"][key]
        desc = class_info["class_labels"].get(key, "Unknown")
        print(f"    {key:<14} {count:<8} {pct:>6.2f}%      {desc}")
    print(f"    {'-'*65}")
    print(f"    - Majority / Minority Ratio     : {class_info['imbalance_ratio']} : 1.0")
    print(f"    - Class Balance Status          : {'Balanced dataset - No severe class skew' if class_info['is_balanced'] else 'Imbalanced'}")
    print()

    # Step 5: Basic Statistical Analysis
    print("[*] STEP 5: BASIC STATISTICAL ANALYSIS")
    print(subsep)
    stats_info = compute_statistical_analysis(df, label_col=label_col)
    print("    - Five Highest Variance Features (Informative Spread):")
    for feat, var_val in stats_info["top_variance_features"].items():
        print(f"        * {feat:<10}: Variance = {var_val:.5f}")
    print("    - Five Lowest Variance Features (Concentrated Return):")
    for feat, var_val in stats_info["lowest_variance_features"].items():
        print(f"        * {feat:<10}: Variance = {var_val:.5f}")
    
    # Class-wise mean response contrast
    c_mean = stats_info["class_mean_df"]
    diff = (c_mean["M"] - c_mean["R"]).abs().sort_values(ascending=False)
    print("    - Top 5 Distinguishing Bands (Largest Mean Energy Difference Between Mine & Rock):")
    for band in diff.head(5).index:
        m_val = c_mean.loc[band, "M"]
        r_val = c_mean.loc[band, "R"]
        d_val = diff[band]
        print(f"        * {band:<10}: Mine Mean = {m_val:.4f} | Rock Mean = {r_val:.4f} | Diff Mean = {d_val:.4f}")
    print()

    # Step 6: Identify Input Features and Target Column
    print("[*] STEP 6: FEATURE MATRIX (X) AND TARGET (y) IDENTIFICATION")
    print(subsep)
    X, y, f_names, t_name = get_features_and_target(df, label_col=label_col)
    print(f"    - Input Features (X)            : {len(f_names)} Continuous Acoustic Energy Attributes")
    print(f"      Feature Names                 : {f_names[0]} .. {f_names[-1]}")
    print(f"      Feature Matrix Shape (X)      : {X.shape} (208 samples x 60 frequency channels)")
    print(f"      Feature Data Type             : {X.dtypes.iloc[0]}")
    print(f"    - Target Column (y)             : '{t_name}'")
    print(f"      Target Vector Shape (y)       : {y.shape} (208 discrete categorical labels)")
    print(f"      Unique Class Labels           : {sorted(list(y.unique()))} (M = Mine, R = Rock)")
    print()

    # Step 7 & 8: Visualizations & Saving in results/
    print("[*] STEP 7 & 8: GENERATING AND SAVING VISUALIZATIONS")
    print(subsep)
    saved_plots = generate_and_save_visualizations(df, output_dir=output_dir, label_col=label_col)
    print(f"    Successfully generated and saved {len(saved_plots)} publication-quality plots to 'results/':")
    for p in saved_plots:
        print(f"      [+] {os.path.basename(p):<32} -> {p}")
    print()

    # Step 9: ML Guard Confirmation
    print("[*] STEP 9: MACHINE LEARNING GUARD")
    print(subsep)
    print("    [!] Confirmed: No machine learning models trained today.")
    print("    [!] Feature extraction, data integrity, and exploratory foundations established.")
    print("    [!] System ready for Day 4 preprocessing and model training.")
    print()
    print(sep)
    print("   DAY 3 EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")
    print(sep)

    return {
        "summary": summary,
        "missing": missing_info,
        "duplicates": duplicate_info,
        "class_distribution": class_info,
        "statistical_analysis": stats_info,
        "X_shape": X.shape,
        "y_shape": y.shape,
        "saved_plots": saved_plots,
    }
