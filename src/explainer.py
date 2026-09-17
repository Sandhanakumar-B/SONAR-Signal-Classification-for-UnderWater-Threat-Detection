"""
src/explainer.py
----------------
Model Explainability, Attribution & Interpretability Engine for the
Sonar Signal Classification System (Day 8).

Capabilities:
  1. Permutation Feature Importance:
     - Model-agnostic, test-set-grounded feature importance.
     - Quantifies the drop in F1-score / accuracy when feature values are permuted.
  2. SHAP (SHapley Additive exPlanations) Analysis:
     - Model-agnostic KernelExplainer / TreeExplainer attribution.
     - Summary bar plot (mean absolute SHAP values across acoustic frequency bands).
     - Beeswarm / dot plot revealing directional impact of high vs. low feature values on Mine prediction.
     - Local instance force/waterfall explanation for representative high-confidence threat and rock samples.
  3. Misclassification & Boundary Case Analysis:
     - Isolates False Positives (FP - benign rocks flagged as mines) and False Negatives (FN - missed mines).
     - Computes spectral error profiles comparing misclassified signals with typical mine/rock signatures.
  4. Publication-quality figure generation & JSON artifact persistence to results/.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report

# Canonical directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def load_explainability_data():
    """
    Load preprocessed test and train datasets, labels, and the champion model.

    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, model, scaler, feature_names)
    """
    X_train_path = os.path.join(PROCESSED_DATA_DIR, "X_train.csv")
    X_test_path = os.path.join(PROCESSED_DATA_DIR, "X_test.csv")
    y_train_path = os.path.join(PROCESSED_DATA_DIR, "y_train.csv")
    y_test_path = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")
    model_path = os.path.join(MODELS_DIR, "best_tuned_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    meta_path = os.path.join(PROCESSED_DATA_DIR, "preprocessing_metadata.json")

    if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path, model_path]):
        raise FileNotFoundError(
            "Required data or model artifacts missing. Ensure preprocess.py and tune.py have been executed."
        )

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    if hasattr(model, "feature_names_in_"):
        raw_feature_names = list(model.feature_names_in_)
    elif os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            raw_feature_names = meta.get("feature_names", list(X_train.columns))
    else:
        raw_feature_names = list(X_train.columns)

    X_train.columns = raw_feature_names
    X_test.columns = raw_feature_names

    # Clean display names for visualizations (e.g. Freq_01, Freq_02...)
    display_feature_names = [f"Freq_{i+1:02d}" for i in range(len(raw_feature_names))]

    return X_train, y_train, X_test, y_test, model, scaler, display_feature_names


def compute_permutation_importance(
    model,
    X_test,
    y_test,
    feature_names,
    n_repeats=15,
    random_state=42,
    scoring="f1"
):
    """
    Compute permutation feature importance on unseen test data.

    Parameters
    ----------
    model : fitted classifier
    X_test, y_test : test set
    feature_names : list of str
    n_repeats : int, default=15
    random_state : int, default=42
    scoring : str, default='f1'

    Returns
    -------
    dict
        Structured permutation importance records and raw result object.
    """
    print(f"[*] Calculating Permutation Feature Importance ({scoring.upper()} metric, {n_repeats} shuffles)...")
    perm_res = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )

    records = []
    for i, name in enumerate(feature_names):
        records.append({
            "feature": name,
            "mean_importance": float(perm_res.importances_mean[i]),
            "std_importance": float(perm_res.importances_std[i]),
        })

    records.sort(key=lambda x: x["mean_importance"], reverse=True)
    return {
        "records": records,
        "raw": perm_res
    }


def compute_shap_explanations(
    model,
    X_train,
    X_test,
    feature_names,
    nsamples=100,
    background_size=40,
    random_state=42
):
    """
    Compute model-agnostic SHAP values using KernelExplainer (or TreeExplainer where applicable).

    Parameters
    ----------
    model : fitted classifier
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    feature_names : list of str
    nsamples : int
    background_size : int
    random_state : int

    Returns
    -------
    dict
        SHAP values, base value, background sample, and summary statistics.
    """
    import shap

    print("[*] Initializing SHAP Explainer...")
    np.random.seed(random_state)

    # Use a stratified or representative background sample from X_train for speed
    if len(X_train) > background_size:
        background_idx = np.random.choice(len(X_train), size=background_size, replace=False)
        background = X_train.iloc[background_idx]
    else:
        background = X_train

    # Prediction function for probability of Mine (Class 1)
    if hasattr(model, "predict_proba"):
        def predict_fn(x):
            # Maintain DataFrame structure if needed or pass values cleanly
            if isinstance(x, np.ndarray) and hasattr(model, "feature_names_in_"):
                x_df = pd.DataFrame(x, columns=model.feature_names_in_)
                return model.predict_proba(x_df)[:, 1]
            return model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        def predict_fn(x):
            if isinstance(x, np.ndarray) and hasattr(model, "feature_names_in_"):
                x_df = pd.DataFrame(x, columns=model.feature_names_in_)
                return model.decision_function(x_df)
            return model.decision_function(x)
    else:
        def predict_fn(x):
            if isinstance(x, np.ndarray) and hasattr(model, "feature_names_in_"):
                x_df = pd.DataFrame(x, columns=model.feature_names_in_)
                return model.predict(x_df)
            return model.predict(x)

    print(f"[*] Computing SHAP values across test samples (background size={len(background)}, nsamples={nsamples})...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_test, nsamples=nsamples, silent=True)

    # If returned as a list of classes, pick class 1 (Mine)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values

    # Mean absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_values_class1), axis=0)
    feature_shap_rank = []
    for i, name in enumerate(feature_names):
        feature_shap_rank.append({
            "feature": name,
            "mean_abs_shap": float(mean_abs_shap[i]),
        })
    feature_shap_rank.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    return {
        "explainer": explainer,
        "shap_values": shap_values_class1,
        "expected_value": float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]),
        "feature_shap_rank": feature_shap_rank,
        "background": background
    }


def analyze_misclassifications(model, X_test, y_test, feature_names, threshold=0.43):
    """
    Isolate and deeply profile misclassified samples (False Positives and False Negatives).

    Parameters
    ----------
    model : fitted classifier
    X_test : pd.DataFrame
    y_test : array-like
    feature_names : list of str
    threshold : float, default=0.43 (Day 6 safe threshold)

    Returns
    -------
    dict
        Misclassification index lists, details, and spectral comparisons.
    """
    print(f"[*] Analyzing Decision Boundary & Misclassifications (Threshold = {threshold:.2f})...")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test).astype(float)

    preds = (probs >= threshold).astype(int)

    tp_indices = []
    tn_indices = []
    fp_indices = []
    fn_indices = []

    for i in range(len(y_test)):
        actual = int(y_test[i])
        predicted = int(preds[i])
        if actual == 1 and predicted == 1:
            tp_indices.append(i)
        elif actual == 0 and predicted == 0:
            tn_indices.append(i)
        elif actual == 0 and predicted == 1:
            fp_indices.append(i)
        elif actual == 1 and predicted == 0:
            fn_indices.append(i)

    misclassified_records = []
    for idx in fp_indices:
        misclassified_records.append({
            "test_sample_index": int(idx),
            "actual": "Rock",
            "predicted": "Mine",
            "error_type": "False Positive (False Alarm)",
            "predicted_prob_mine": float(probs[idx]),
            "margin_from_threshold": float(probs[idx] - threshold)
        })

    for idx in fn_indices:
        misclassified_records.append({
            "test_sample_index": int(idx),
            "actual": "Mine",
            "predicted": "Rock",
            "error_type": "False Negative (LETHAL MISSED MINE)",
            "predicted_prob_mine": float(probs[idx]),
            "margin_from_threshold": float(threshold - probs[idx])
        })

    print(f"    - True Positives  (Mines detected) : {len(tp_indices)}")
    print(f"    - True Negatives  (Rocks cleared)  : {len(tn_indices)}")
    print(f"    - False Positives (False Alarms)   : {len(fp_indices)}")
    print(f"    - False Negatives (Missed Mines)   : {len(fn_indices)}")

    return {
        "threshold": float(threshold),
        "tp_indices": tp_indices,
        "tn_indices": tn_indices,
        "fp_indices": fp_indices,
        "fn_indices": fn_indices,
        "misclassified_records": misclassified_records,
        "probs": probs.tolist(),
        "preds": preds.tolist()
    }


def render_and_save_explainability_plots(
    perm_output,
    shap_output,
    misclass_output,
    X_test,
    y_test,
    output_dir=None
):
    """
    Render publication-quality interpretability visualisations to results/.

    Plots:
      1. results/permutation_importance.png (Top 15 bands by permutation score degradation)
      2. results/shap_summary.png (Mean |SHAP| global importance bar chart)
      3. results/shap_beeswarm.png (Beeswarm directional impact chart)
      4. results/misclassification_analysis.png (Spectral profiles of errors vs correct predictions)
      5. results/local_prediction_waterfall.png (Waterfall explanations of high confidence samples)

    Returns
    -------
    dict
        Paths to saved plot images.
    """
    target_dir = output_dir if output_dir is not None else RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    saved_plots = {}

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    # =========================================================================
    # PLOT 1: Permutation Feature Importance (Top 15)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    top_perm = perm_output["records"][:15]
    p_names = [r["feature"] for r in top_perm][::-1]
    p_means = [r["mean_importance"] for r in top_perm][::-1]
    p_stds = [r["std_importance"] for r in top_perm][::-1]

    colors = ["#2b5c8f" if m > 0 else "#a83232" for m in p_means]
    bars = ax.barh(p_names, p_means, xerr=p_stds, color=colors, alpha=0.85, edgecolor="black", capsize=3)

    for bar in bars:
        w = bar.get_width()
        x_pos = w + 0.005 if w >= 0 else w - 0.015
        ax.annotate(
            f"{w:.4f}",
            xy=(x_pos, bar.get_y() + bar.get_height() / 2),
            va="center",
            fontsize=8.5,
            fontweight="bold"
        )

    ax.set_title("Test-Set Permutation Feature Importance (F1 Degradation on Shuffle)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Δ F1-Score (Higher indicates more critical acoustic band)", fontsize=10, fontweight="bold")
    plt.tight_layout()

    path_perm = os.path.join(target_dir, "permutation_importance.png")
    fig.savefig(path_perm)
    plt.close(fig)
    saved_plots["permutation_importance"] = path_perm

    # =========================================================================
    # PLOT 2: SHAP Global Feature Importance Bar Chart (Top 15)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    top_shap = shap_output["feature_shap_rank"][:15]
    s_names = [r["feature"] for r in top_shap][::-1]
    s_vals = [r["mean_abs_shap"] for r in top_shap][::-1]

    bars = ax.barh(s_names, s_vals, color="#e67e22", alpha=0.88, edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.4f}",
            xy=(w + (max(s_vals)*0.01), bar.get_y() + bar.get_height() / 2),
            va="center",
            fontsize=8.5,
            fontweight="bold"
        )

    ax.set_title("Global Feature Attribution: Mean |SHAP| Value (Impact on Mine Prediction)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value across Test Signals", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(s_vals) * 1.18)
    plt.tight_layout()

    path_shap_bar = os.path.join(target_dir, "shap_summary.png")
    fig.savefig(path_shap_bar)
    plt.close(fig)
    saved_plots["shap_summary"] = path_shap_bar

    # =========================================================================
    # PLOT 3: SHAP Beeswarm Summary Plot
    # =========================================================================
    import shap
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    plt.sca(ax)
    
    # Render beeswarm with custom formatting
    shap.summary_plot(
        shap_output["shap_values"],
        X_test,
        feature_names=X_test.columns.tolist(),
        max_display=15,
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP Beeswarm: Feature Value Impact on Naval Threat Likelihood", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    path_shap_bee = os.path.join(target_dir, "shap_beeswarm.png")
    fig.savefig(path_shap_bee)
    plt.close(fig)
    saved_plots["shap_beeswarm"] = path_shap_bee

    # =========================================================================
    # PLOT 4: Misclassification & Spectral Signature Deviation Analysis
    # =========================================================================
    fig, (ax_spec, ax_margins) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    # 4A: Spectral Curves (Mean True Mines vs True Rocks vs Misclassified)
    tp_idx = misclass_output["tp_indices"]
    tn_idx = misclass_output["tn_indices"]
    fp_idx = misclass_output["fp_indices"]
    fn_idx = misclass_output["fn_indices"]

    freq_indices = np.arange(1, 61)

    if len(tp_idx) > 0:
        mean_tp_spec = X_test.iloc[tp_idx].mean(axis=0).values
        ax_spec.plot(freq_indices, mean_tp_spec, label=f"True Mines (TP, n={len(tp_idx)})", color="#d62728", lw=2)
    if len(tn_idx) > 0:
        mean_tn_spec = X_test.iloc[tn_idx].mean(axis=0).values
        ax_spec.plot(freq_indices, mean_tn_spec, label=f"True Rocks (TN, n={len(tn_idx)})", color="#1f77b4", lw=2)

    # Overlay any False Positives
    if len(fp_idx) > 0:
        for f_i, idx in enumerate(fp_idx):
            ax_spec.plot(
                freq_indices,
                X_test.iloc[idx].values,
                linestyle="--",
                alpha=0.75,
                color="#e67e22",
                label="FP (Rock predicted as Mine)" if f_i == 0 else None
            )
    # Overlay any False Negatives
    if len(fn_idx) > 0:
        for f_i, idx in enumerate(fn_idx):
            ax_spec.plot(
                freq_indices,
                X_test.iloc[idx].values,
                linestyle=":",
                alpha=0.85,
                color="#9b59b6",
                label="FN (LETHAL MISSED MINE)" if f_i == 0 else None
            )

    ax_spec.set_title("Spectral Profile: Correct vs. Misclassified Signals", fontsize=11, fontweight="bold")
    ax_spec.set_xlabel("Acoustic Frequency Band (1 to 60)", fontsize=9, fontweight="bold")
    ax_spec.set_ylabel("Standardized Acoustic Energy", fontsize=9, fontweight="bold")
    ax_spec.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # 4B: Prediction Probability & Threshold Safety Margin
    probs = np.array(misclass_output["probs"])
    thresh = misclass_output["threshold"]

    # Scatter plot of all test samples sorted by probability
    sorted_order = np.argsort(probs)
    sorted_probs = probs[sorted_order]
    sorted_labels = y_test[sorted_order]

    colors_scatter = ["#1f77b4" if l == 0 else "#d62728" for l in sorted_labels]
    ax_margins.scatter(np.arange(len(sorted_probs)), sorted_probs, c=colors_scatter, s=45, edgecolor="black", alpha=0.9, label="Test Samples")
    ax_margins.axhline(y=thresh, color="darkgreen", linestyle="--", lw=2, label=f"Safe Threshold (tau={thresh:.2f})")
    ax_margins.axhline(y=0.50, color="gray", linestyle=":", lw=1.5, label="Default Threshold (0.50)")

    # Highlight error samples
    for i, idx in enumerate(sorted_order):
        if idx in fp_idx or idx in fn_idx:
            ax_margins.scatter(i, sorted_probs[i], s=130, facecolors="none", edgecolors="red", lw=2.5)

    ax_margins.set_title(f"Threat Likelihood Distribution & Safety Margin ({len(misclass_output['misclassified_records'])} errors)", fontsize=11, fontweight="bold")
    ax_margins.set_xlabel("Sorted Test Sample Index", fontsize=9, fontweight="bold")
    ax_margins.set_ylabel("Predicted Mine Probability", fontsize=9, fontweight="bold")
    ax_margins.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    path_misclass = os.path.join(target_dir, "misclassification_analysis.png")
    fig.savefig(path_misclass)
    plt.close(fig)
    saved_plots["misclassification_analysis"] = path_misclass

    # =========================================================================
    # PLOT 5: Local Sample Attribution Waterfalls (Mine vs Rock)
    # =========================================================================
    fig, (ax_mine, ax_rock) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Identify most confident Mine (highest prob) and Rock (lowest prob)
    most_confident_mine_idx = int(np.argmax(probs))
    most_confident_rock_idx = int(np.argmin(probs))

    # Top 8 contributing features for the mine
    mine_shaps = shap_output["shap_values"][most_confident_mine_idx]
    top_mine_feat_idx = np.argsort(np.abs(mine_shaps))[-8:]
    mine_names = [X_test.columns[i] for i in top_mine_feat_idx]
    mine_vals = [mine_shaps[i] for i in top_mine_feat_idx]
    mine_colors = ["#27ae60" if v > 0 else "#c0392b" for v in mine_vals]

    ax_mine.barh(mine_names, mine_vals, color=mine_colors, alpha=0.85, edgecolor="black")
    ax_mine.axvline(0, color="black", lw=1)
    ax_mine.set_title(f"Local Attribution: Confident Mine (Prob={probs[most_confident_mine_idx]:.3f})", fontsize=11, fontweight="bold")
    ax_mine.set_xlabel("SHAP Contribution (Green = Pushes to Mine)", fontsize=9, fontweight="bold")

    # Top 8 contributing features for the rock
    rock_shaps = shap_output["shap_values"][most_confident_rock_idx]
    top_rock_feat_idx = np.argsort(np.abs(rock_shaps))[-8:]
    rock_names = [X_test.columns[i] for i in top_rock_feat_idx]
    rock_vals = [rock_shaps[i] for i in top_rock_feat_idx]
    rock_colors = ["#27ae60" if v > 0 else "#c0392b" for v in rock_vals]

    ax_rock.barh(rock_names, rock_vals, color=rock_colors, alpha=0.85, edgecolor="black")
    ax_rock.axvline(0, color="black", lw=1)
    ax_rock.set_title(f"Local Attribution: Confident Rock (Prob={probs[most_confident_rock_idx]:.3f})", fontsize=11, fontweight="bold")
    ax_rock.set_xlabel("SHAP Contribution (Red = Pushes to Rock)", fontsize=9, fontweight="bold")

    plt.suptitle("Local Instance Explanations: Key Frequencies Driving Specific Predictions", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    path_local = os.path.join(target_dir, "local_prediction_waterfall.png")
    fig.savefig(path_local)
    plt.close(fig)
    saved_plots["local_prediction_waterfall"] = path_local

    print(f"\n[*] Generated & Saved 5 Interpretability Visualizations to '{target_dir}':")
    for k, p in saved_plots.items():
        print(f"    - {k:<28} -> {os.path.basename(p)}")

    return saved_plots


def save_explainability_artifacts(
    perm_output,
    shap_output,
    misclass_output,
    results_dir=None
):
    """
    Export structured explainability findings to results/explainability_report.json.

    Returns
    -------
    str
        Path to JSON report.
    """
    target_results = results_dir if results_dir is not None else RESULTS_DIR
    os.makedirs(target_results, exist_ok=True)

    report_payload = {
        "timestamp": datetime.now().isoformat(),
        "model_analyzed": "Support Vector Machine (best_tuned_model.joblib)",
        "safe_decision_threshold": misclass_output["threshold"],
        "top_permutation_features": perm_output["records"][:15],
        "top_shap_features": shap_output["feature_shap_rank"][:15],
        "misclassifications_summary": {
            "total_test_samples": len(misclass_output["probs"]),
            "true_positives": len(misclass_output["tp_indices"]),
            "true_negatives": len(misclass_output["tn_indices"]),
            "false_positives": len(misclass_output["fp_indices"]),
            "false_negatives": len(misclass_output["fn_indices"]),
            "zero_missed_mines_verified": (len(misclass_output["fn_indices"]) == 0),
            "errors": misclass_output["misclassified_records"]
        }
    }

    report_path = os.path.join(target_results, "explainability_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)

    print(f"\n[*] Persisted Explainability Report to: {os.path.basename(report_path)}")
    return report_path


def run_full_explainability(
    nsamples=100,
    background_size=40,
    random_state=42,
    output_dir=None
):
    """
    Main orchestration entry point for Day 8 Model Explainability & Interpretability.
    """
    print("\n" + "=" * 76)
    print("   DAY 8: MODEL EXPLAINABILITY, SHAP & ERROR PROFILE ANALYSIS")
    print("=" * 76)

    # 1. Load data and models
    X_train, y_train, X_test, y_test, model, scaler, feature_names = load_explainability_data()

    # Load threshold from diagnostics
    diag_path = os.path.join(RESULTS_DIR, "tuning_and_diagnostics.json")
    safe_thresh = 0.43
    if os.path.exists(diag_path):
        try:
            with open(diag_path, "r") as f:
                d = json.load(f)
                safe_thresh = d.get("operating_points", {}).get("high_recall_safe", {}).get("threshold", 0.43)
        except Exception:
            pass

    # 2. Permutation Importance
    perm_output = compute_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        n_repeats=15,
        random_state=random_state,
        scoring="f1"
    )

    # 3. SHAP Analysis
    shap_output = compute_shap_explanations(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        nsamples=nsamples,
        background_size=background_size,
        random_state=random_state
    )

    # 4. Misclassification Analysis
    misclass_output = analyze_misclassifications(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        threshold=safe_thresh
    )

    # 5. Render Plots
    saved_plots = render_and_save_explainability_plots(
        perm_output=perm_output,
        shap_output=shap_output,
        misclass_output=misclass_output,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir
    )

    # 6. Save JSON Report
    report_path = save_explainability_artifacts(
        perm_output=perm_output,
        shap_output=shap_output,
        misclass_output=misclass_output,
        results_dir=output_dir
    )

    return {
        "permutation": perm_output,
        "shap": shap_output,
        "misclassifications": misclass_output,
        "plots": saved_plots,
        "report_path": report_path
    }
