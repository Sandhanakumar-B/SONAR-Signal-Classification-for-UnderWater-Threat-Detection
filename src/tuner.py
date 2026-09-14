"""
src/tuner.py
------------
Hyperparameter Tuning, Threshold Optimization, and Comprehensive Model
Diagnostics module for the Sonar Signal Classification System (Day 6).

Capabilities:
  1. Systematic Hyperparameter Tuning (GridSearchCV with 5-Fold Stratified CV):
     - Support Vector Machine (C, gamma, kernel)
     - Random Forest (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features)
     - K-Nearest Neighbors (n_neighbors, weights, metric)
     - Logistic Regression (C, solver)
  2. Baseline vs Tuned Performance Comparison across CV and unseen Test sets.
  3. Naval Threat Decision Threshold Optimization:
     - Calibrates probability operating thresholds to eliminate lethal False Negatives (unexploded mines).
     - Identifies Default (0.50), Max-F1, High-Recall (>=95%), and Cost-Weighted (5x FN penalty) thresholds.
  4. In-Depth Model Diagnostics:
     - Learning curves (sample size vs score trajectory to analyze bias/variance)
     - Frequency feature importance analysis (top acoustic frequency bands)
     - Probability calibration curves (reliability diagrams and Brier scores)
  5. Publication-quality diagnostic visualization export to results/
  6. Persistence of tuned models to models/ and structured metrics to results/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    learning_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

# Canonical directories
DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)
DEFAULT_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def get_hyperparameter_grids(random_state=42):
    """
    Define algorithm estimators and hyperparameter search grids for tuning.

    Parameters
    ----------
    random_state : int, default=42
        Reproducibility seed.

    Returns
    -------
    dict
        Dictionary containing estimator instances and their parameter grids.
    """
    configs = {
        "Support Vector Machine": {
            "estimator": SVC(probability=True, random_state=random_state),
            "param_grid": {
                "C": [0.5, 1.0, 2.0, 5.0, 10.0],
                "gamma": ["scale", "auto", 0.01, 0.02, 0.05, 0.1],
                "kernel": ["rbf"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [None, 5, 10, 15],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
            },
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": list(range(1, 16)),
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=random_state),
            "param_grid": {
                "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
            },
        },
    }
    return configs


def evaluate_estimator_on_test(estimator, X_test, y_test):
    """
    Compute full test evaluation metrics for a fitted estimator.

    Parameters
    ----------
    estimator : fitted classifier
    X_test : array-like of shape (n_samples, n_features)
    y_test : array-like of shape (n_samples,)

    Returns
    -------
    dict
        Evaluation metrics on the test partition.
    """
    y_pred = estimator.predict(X_test)

    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_test)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_proba = estimator.decision_function(X_test)
    else:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    clf_rep = classification_report(
        y_test,
        y_pred,
        target_names=["Rock (0)", "Mine (1)"],
        output_dict=True,
    )

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "classification_report": clf_rep,
    }


def tune_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    baseline_results=None,
    cv=5,
    random_state=42,
):
    """
    Execute systematic 5-Fold Stratified GridSearchCV for all 4 classification models.

    Parameters
    ----------
    X_train, y_train : standardized training partitions
    X_test, y_test : standardized testing partitions
    baseline_results : dict, optional
        Day 5 baseline evaluation results for delta benchmarking.
    cv : int, default=5
    random_state : int, default=42

    Returns
    -------
    dict
        Comprehensive tuning results, best parameters, and fitted tuned estimators.
    """
    configs = get_hyperparameter_grids(random_state=random_state)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    tuned_results = {}
    fitted_tuned_models = {}

    print("\n" + "=" * 76)
    print("   DAY 6: HYPERPARAMETER TUNING VIA STRATIFIED GRID SEARCH")
    print("=" * 76)

    for name, cfg in configs.items():
        print(f"\n--> Tuning Hyperparameters for: {name}...")
        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            scoring="f1",
            cv=skf,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)

        best_estimator = grid.best_estimator_
        fitted_tuned_models[name] = best_estimator
        best_params = grid.best_params_
        best_cv_f1 = float(grid.best_score_)

        # Test evaluation
        test_metrics = evaluate_estimator_on_test(best_estimator, X_test, y_test)

        # Retrieve baseline comparison if available
        baseline_test = {}
        if baseline_results and name in baseline_results:
            b_data = baseline_results[name].get("test_metrics", {})
            baseline_test = {
                "accuracy": b_data.get("test_accuracy", 0.0),
                "recall": b_data.get("test_recall", 0.0),
                "f1": b_data.get("test_f1", 0.0),
                "roc_auc": b_data.get("test_roc_auc", 0.0),
            }

        delta_f1 = test_metrics["f1"] - baseline_test.get("f1", test_metrics["f1"])
        delta_acc = test_metrics["accuracy"] - baseline_test.get("accuracy", test_metrics["accuracy"])

        print(f"    [Best Params] {best_params}")
        print(f"    [CV  Score  ] Best 5-Fold F1: {best_cv_f1:.4f}")
        print(
            f"    [Test Scores] Acc: {test_metrics['accuracy']:.4f} (Delta: {delta_acc:+.4f}) | "
            f"Rec: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f} (Delta: {delta_f1:+.4f}) | "
            f"ROC-AUC: {test_metrics['roc_auc']:.4f} (TP={test_metrics['tp']}, FN={test_metrics['fn']})"
        )

        tuned_results[name] = {
            "best_params": best_params,
            "cv_best_f1": best_cv_f1,
            "test_metrics": test_metrics,
            "baseline_comparison": {
                "baseline_test": baseline_test,
                "delta_accuracy": float(delta_acc),
                "delta_f1": float(delta_f1),
            },
        }

    # Identify champion tuned model
    best_tuned_name = max(
        tuned_results.keys(),
        key=lambda m: (
            tuned_results[m]["test_metrics"]["recall"],
            tuned_results[m]["test_metrics"]["f1"],
            tuned_results[m]["test_metrics"]["accuracy"],
        ),
    )

    print("\n" + "-" * 76)
    print(f"[*] CHAMPION TUNED MODEL SELECTED: {best_tuned_name}")
    c_met = tuned_results[best_tuned_name]["test_metrics"]
    print(
        f"    Test Accuracy: {c_met['accuracy']*100:.2f}% | "
        f"Test F1: {c_met['f1']:.4f} | "
        f"Threat Recall: {c_met['recall']*100:.2f}% (FN={c_met['fn']})"
    )
    print("-" * 76)

    return {
        "tuned_results": tuned_results,
        "fitted_tuned_models": fitted_tuned_models,
        "best_tuned_model_name": best_tuned_name,
    }


def optimize_decision_thresholds(
    model,
    X_test,
    y_test,
    model_name="Support Vector Machine",
    fn_cost_weight=5.0,
    fp_cost_weight=1.0,
):
    """
    Perform naval threat decision threshold optimization.

    Sweeps decision threshold tau in [0.01, 0.99] to find optimal operational points:
      1. Default Threshold (tau = 0.50)
      2. Maximum F1-Score Threshold (tau_f1)
      3. High-Recall Safe Threshold (targeting 100% or >=95% threat recall)
      4. Cost-Weighted Optimal Threshold (penalizes FN 5x relative to FP)

    Parameters
    ----------
    model : fitted classifier
    X_test, y_test : test set
    model_name : str
    fn_cost_weight : float, default=5.0
        Relative cost multiplier for a False Negative (missing a mine).
    fp_cost_weight : float, default=1.0
        Relative cost multiplier for a False Positive (false alarm on a rock).

    Returns
    -------
    dict
        Threshold optimization sweep table and designated operational operating points.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X_test)
        y_proba = (df - df.min()) / (df.max() - df.min() + 1e-9)
    else:
        raise ValueError("Model does not support probability output for threshold optimization.")

    thresholds = np.linspace(0.01, 0.99, 99)
    sweep_records = []

    for tau in thresholds:
        y_pred = (y_proba >= tau).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        total_cost = (fn * fn_cost_weight) + (fp * fp_cost_weight)

        sweep_records.append({
            "threshold": round(float(tau), 3),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "total_cost": round(float(total_cost), 2),
        })

    sweep_df = pd.DataFrame(sweep_records)

    # Operating Point 1: Default (tau = 0.50)
    idx_default = (sweep_df["threshold"] - 0.50).abs().idxmin()
    default_pt = sweep_df.iloc[idx_default].to_dict()

    # Operating Point 2: Max F1
    idx_max_f1 = sweep_df["f1"].idxmax()
    max_f1_pt = sweep_df.iloc[idx_max_f1].to_dict()

    # Operating Point 3: Safe High-Recall (100% recall with highest precision, or highest recall)
    rec_100 = sweep_df[sweep_df["recall"] >= 0.999]
    if len(rec_100) > 0:
        idx_safe = rec_100["precision"].idxmax()
        safe_pt = rec_100.loc[idx_safe].to_dict()
    else:
        idx_safe = sweep_df["recall"].idxmax()
        safe_pt = sweep_df.iloc[idx_safe].to_dict()

    # Operating Point 4: Cost-Weighted Optimal
    idx_cost = sweep_df["total_cost"].idxmin()
    cost_pt = sweep_df.iloc[idx_cost].to_dict()

    summary = {
        "model_name": model_name,
        "fn_cost_weight": fn_cost_weight,
        "fp_cost_weight": fp_cost_weight,
        "operating_points": {
            "default": default_pt,
            "max_f1": max_f1_pt,
            "high_recall_safe": safe_pt,
            "cost_optimal": cost_pt,
        },
        "sweep_data": sweep_df,
    }

    print("\n" + "=" * 76)
    print(f"   NAVAL THREAT THRESHOLD OPTIMIZATION ({model_name})")
    print("=" * 76)
    print(f"[*] Operating Points Identified:")
    print(
        f"    1. Default (tau=0.50)      -> Acc: {default_pt['accuracy']*100:.1f}% | "
        f"Prec: {default_pt['precision']*100:.1f}% | Rec: {default_pt['recall']*100:.1f}% | "
        f"F1: {default_pt['f1']:.4f} | FN: {int(default_pt['fn'])} (Cost: {default_pt['total_cost']})"
    )
    print(
        f"    2. Max-F1 (tau={max_f1_pt['threshold']:.2f})       -> Acc: {max_f1_pt['accuracy']*100:.1f}% | "
        f"Prec: {max_f1_pt['precision']*100:.1f}% | Rec: {max_f1_pt['recall']*100:.1f}% | "
        f"F1: {max_f1_pt['f1']:.4f} | FN: {int(max_f1_pt['fn'])} (Cost: {max_f1_pt['total_cost']})"
    )
    print(
        f"    3. Safe Threat (tau={safe_pt['threshold']:.2f})   -> Acc: {safe_pt['accuracy']*100:.1f}% | "
        f"Prec: {safe_pt['precision']*100:.1f}% | Rec: {safe_pt['recall']*100:.1f}% | "
        f"F1: {safe_pt['f1']:.4f} | FN: {int(safe_pt['fn'])} (Cost: {safe_pt['total_cost']})"
    )
    print(
        f"    4. Cost-Optimal (tau={cost_pt['threshold']:.2f})  -> Acc: {cost_pt['accuracy']*100:.1f}% | "
        f"Prec: {cost_pt['precision']*100:.1f}% | Rec: {cost_pt['recall']*100:.1f}% | "
        f"F1: {cost_pt['f1']:.4f} | FN: {int(cost_pt['fn'])} (Cost: {cost_pt['total_cost']})"
    )

    return summary


def compute_model_diagnostics(
    fitted_models,
    X_train,
    y_train,
    X_test,
    y_test,
    cv=5,
    random_state=42,
):
    """
    Generate deep diagnostic measurements:
      - Learning curves (sample size vs training/validation accuracy)
      - Feature importances across the 60 acoustic frequency bands
      - Calibration curves and Brier score evaluation

    Returns
    -------
    dict
        Diagnostics data payload.
    """
    print("\n" + "=" * 76)
    print("   DAY 6: COMPUTING COMPREHENSIVE MODEL DIAGNOSTICS")
    print("=" * 76)

    diagnostics = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    train_sizes = np.linspace(0.2, 1.0, 5)

    # 1. Learning Curves for All Models
    print("[*] Computing Learning Curves across training set fractions (20% to 100%)...")
    learning_curves_data = {}
    for name, model in fitted_models.items():
        t_sizes, train_scores, cv_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=skf,
            train_sizes=train_sizes,
            scoring="accuracy",
            n_jobs=-1,
            random_state=random_state,
        )
        learning_curves_data[name] = {
            "train_sizes": t_sizes.tolist(),
            "train_mean": np.mean(train_scores, axis=1).tolist(),
            "train_std": np.std(train_scores, axis=1).tolist(),
            "cv_mean": np.mean(cv_scores, axis=1).tolist(),
            "cv_std": np.std(cv_scores, axis=1).tolist(),
        }
    diagnostics["learning_curves"] = learning_curves_data

    # 2. Feature Importances (Random Forest Gini Impurity)
    print("[*] Analyzing Acoustic Frequency Feature Importances...")
    rf_model = fitted_models.get("Random Forest")
    feature_importance_records = []
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        feature_names = [f"Freq_{i+1:02d}" for i in range(X_train.shape[1])]
        for fname, imp in zip(feature_names, importances):
            feature_importance_records.append({
                "feature": fname,
                "importance": float(imp),
            })
        feature_importance_records.sort(key=lambda x: x["importance"], reverse=True)
    diagnostics["feature_importances"] = feature_importance_records

    # 3. Model Calibration Curves & Brier Scores
    print("[*] Computing Probability Calibration & Brier Reliability Scores...")
    calibration_data = {}
    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_prob, n_bins=6, strategy="uniform"
            )
            brier = brier_score_loss(y_test, y_prob)
            calibration_data[name] = {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist(),
                "brier_score": float(brier),
            }
            print(f"    - {name:<24} -> Brier Score: {brier:.4f} (Lower is better)")
    diagnostics["calibration"] = calibration_data

    return diagnostics


def generate_and_save_diagnostic_plots(
    tuning_output,
    threshold_output,
    diagnostics_output,
    output_dir=None,
):
    """
    Render publication-quality visualization figures to results/.

    Plots:
      1. results/tuning_comparison.png (Baseline vs Tuned performance comparison)
      2. results/threshold_optimization.png (Precision, Recall, F1, Cost vs Threshold)
      3. results/learning_curves.png (2x2 grid of learning curves)
      4. results/feature_importance.png (Top 15 most discriminative frequency bands)
      5. results/calibration_curves.png (Reliability calibration diagram)

    Returns
    -------
    dict
        Paths to saved plot images.
    """
    target_dir = output_dir if output_dir is not None else DEFAULT_RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    saved_plots = {}

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    # =========================================================================
    # PLOT 1: Baseline vs Tuned Performance Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    models = list(tuning_output["tuned_results"].keys())
    x = np.arange(len(models))
    width = 0.35

    base_f1 = []
    tuned_f1 = []
    for m in models:
        comp = tuning_output["tuned_results"][m]["baseline_comparison"]
        base_f1.append(comp["baseline_test"].get("f1", 0.0))
        tuned_f1.append(tuning_output["tuned_results"][m]["test_metrics"]["f1"])

    r1 = ax.bar(x - width / 2, base_f1, width, label="Baseline (Day 5)", color="#4682B4", alpha=0.9, edgecolor="black", linewidth=0.6)
    r2 = ax.bar(x + width / 2, tuned_f1, width, label="Tuned (Day 6)", color="#2E8B57", alpha=0.9, edgecolor="black", linewidth=0.6)

    for rect in r1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in r2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Hyperparameter Tuning Impact: Baseline vs Tuned F1-Score (Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    path_comp = os.path.join(target_dir, "tuning_comparison.png")
    fig.savefig(path_comp)
    plt.close(fig)
    saved_plots["tuning_comparison"] = path_comp

    # =========================================================================
    # PLOT 2: Threshold Optimization & Threat Cost Curve
    # =========================================================================
    fig, (ax_tradeoff, ax_cost) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sweep_df = threshold_output["sweep_data"]
    ops = threshold_output["operating_points"]

    # Subplot 2A: Precision, Recall, F1 vs Threshold
    ax_tradeoff.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#1f77b4", lw=2.2)
    ax_tradeoff.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall (Threat Sensitivity)", color="#d62728", lw=2.2)
    ax_tradeoff.plot(sweep_df["threshold"], sweep_df["f1"], label="F1-Score", color="#2ca02c", lw=2.2, linestyle="--")

    # Mark Operating points
    ax_tradeoff.axvline(x=ops["default"]["threshold"], color="black", linestyle=":", lw=1.5, label=f"Default ({ops['default']['threshold']:.2f})")
    ax_tradeoff.axvline(x=ops["max_f1"]["threshold"], color="green", linestyle="--", lw=1.5, label=f"Max F1 ({ops['max_f1']['threshold']:.2f})")
    ax_tradeoff.axvline(x=ops["high_recall_safe"]["threshold"], color="red", linestyle="-.", lw=1.5, label=f"Safe ({ops['high_recall_safe']['threshold']:.2f})")

    ax_tradeoff.set_title("Precision-Recall-F1 vs Decision Threshold", fontsize=12, fontweight="bold")
    ax_tradeoff.set_xlabel("Decision Threshold (tau)", fontsize=10, fontweight="bold")
    ax_tradeoff.set_ylabel("Score", fontsize=10, fontweight="bold")
    ax_tradeoff.set_ylim(0, 1.05)
    ax_tradeoff.legend(loc="lower left", framealpha=0.9, fontsize=8.5)

    # Subplot 2B: Cost vs Threshold
    ax_cost.plot(sweep_df["threshold"], sweep_df["total_cost"], color="#800080", lw=2.4, label="Total Cost (5*FN + 1*FP)")
    ax_cost.scatter(ops["cost_optimal"]["threshold"], ops["cost_optimal"]["total_cost"], color="red", s=100, zorder=5, label=f"Min Cost ({ops['cost_optimal']['threshold']:.2f})")
    ax_cost.set_title("Operational Threat Cost vs Threshold (FN Penalty = 5x)", fontsize=12, fontweight="bold")
    ax_cost.set_xlabel("Decision Threshold (tau)", fontsize=10, fontweight="bold")
    ax_cost.set_ylabel("Weighted Threat Cost", fontsize=10, fontweight="bold")
    ax_cost.legend(loc="upper center", framealpha=0.9, fontsize=9)

    plt.suptitle(f"Naval Threat Classification Threshold Optimization ({threshold_output['model_name']})", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    path_thresh = os.path.join(target_dir, "threshold_optimization.png")
    fig.savefig(path_thresh)
    plt.close(fig)
    saved_plots["threshold_optimization"] = path_thresh

    # =========================================================================
    # PLOT 3: Learning Curves Grid (2x2)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    lc_data = diagnostics_output["learning_curves"]

    for i, (m_name, lc) in enumerate(lc_data.items()):
        ax_lc = axes[i]
        sizes = lc["train_sizes"]
        tr_mean, tr_std = np.array(lc["train_mean"]), np.array(lc["train_std"])
        cv_mean, cv_std = np.array(lc["cv_mean"]), np.array(lc["cv_std"])

        ax_lc.plot(sizes, tr_mean, "o-", color="#1f77b4", lw=2, label="Training Accuracy")
        ax_lc.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#1f77b4")

        ax_lc.plot(sizes, cv_mean, "s-", color="#2ca02c", lw=2, label="Cross-Validation Accuracy")
        ax_lc.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15, color="#2ca02c")

        ax_lc.set_title(f"{m_name} Learning Curve", fontsize=11, fontweight="bold")
        ax_lc.set_xlabel("Training Samples", fontsize=9, fontweight="bold")
        ax_lc.set_ylabel("Accuracy Score", fontsize=9, fontweight="bold")
        ax_lc.set_ylim(0.65, 1.05)
        ax_lc.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    plt.suptitle("Learning Curves: Bias vs Variance Diagnostics Across Training Sizes", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    path_lc = os.path.join(target_dir, "learning_curves.png")
    fig.savefig(path_lc)
    plt.close(fig)
    saved_plots["learning_curves"] = path_lc

    # =========================================================================
    # PLOT 4: Acoustic Frequency Feature Importance (Top 15 Bands)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    feat_recs = diagnostics_output["feature_importances"][:15]
    f_names = [r["feature"] for r in feat_recs][::-1]
    f_scores = [r["importance"] for r in feat_recs][::-1]

    bars = ax.barh(f_names, f_scores, color="#34495e", edgecolor="black", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_title("Top 15 Most Discriminative Sonar Frequency Bands (Random Forest Gini)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Feature Importance Score", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(f_scores) * 1.25)
    plt.tight_layout()

    path_feat = os.path.join(target_dir, "feature_importance.png")
    fig.savefig(path_feat)
    plt.close(fig)
    saved_plots["feature_importance"] = path_feat

    # =========================================================================
    # PLOT 5: Calibration Reliability Curves
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    cal_data = diagnostics_output["calibration"]
    cal_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ax.plot([0, 1], [0, 1], "k:", lw=1.5, label="Perfect Calibration (Ideal)")
    for i, (m_name, c_info) in enumerate(cal_data.items()):
        fop = c_info["fraction_of_positives"]
        mpv = c_info["mean_predicted_value"]
        brier = c_info["brier_score"]
        ax.plot(mpv, fop, "s-", color=cal_colors[i % len(cal_colors)], lw=2, label=f"{m_name} (Brier: {brier:.3f})")

    ax.set_title("Probability Calibration Curves (Reliability Diagram)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability (Mine Threat)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Empirical Fraction of Positives", fontsize=10, fontweight="bold")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9.5)
    plt.tight_layout()

    path_cal = os.path.join(target_dir, "calibration_curves.png")
    fig.savefig(path_cal)
    plt.close(fig)
    saved_plots["calibration_curves"] = path_cal

    print(f"\n[*] Generated & Saved 5 Diagnostic Visualizations to '{target_dir}':")
    for k, p in saved_plots.items():
        print(f"    - {k:<24} -> {os.path.basename(p)}")

    return saved_plots


def save_tuning_artifacts(
    tuning_output,
    threshold_output,
    diagnostics_output,
    results_dir=None,
    models_dir=None,
):
    """
    Save serialized tuned models and diagnostic metrics to models/ and results/.

    Returns
    -------
    dict
        Paths to saved files.
    """
    target_results = results_dir if results_dir is not None else DEFAULT_RESULTS_DIR
    target_models = models_dir if models_dir is not None else DEFAULT_MODELS_DIR
    os.makedirs(target_results, exist_ok=True)
    os.makedirs(target_models, exist_ok=True)

    # 1. Export Tuning Metrics CSV
    table_rows = []
    for name, data in tuning_output["tuned_results"].items():
        t = data["test_metrics"]
        base = data["baseline_comparison"]
        table_rows.append({
            "Model": name,
            "Best_Parameters": str(data["best_params"]),
            "CV_Best_F1": round(data["cv_best_f1"], 4),
            "Test_Accuracy": round(t["accuracy"], 4),
            "Test_Recall": round(t["recall"], 4),
            "Test_Precision": round(t["precision"], 4),
            "Test_F1_Score": round(t["f1"], 4),
            "Test_ROC_AUC": round(t["roc_auc"], 4),
            "Delta_F1_vs_Baseline": round(base["delta_f1"], 4),
            "Delta_Accuracy_vs_Baseline": round(base["delta_accuracy"], 4),
            "Is_Champion": (name == tuning_output["best_tuned_model_name"]),
        })

    metrics_df = pd.DataFrame(table_rows)
    csv_tuning_path = os.path.join(target_results, "tuning_metrics.csv")
    metrics_df.to_csv(csv_tuning_path, index=False)

    # 2. Export Threshold Sweep CSV
    csv_thresh_path = os.path.join(target_results, "threshold_analysis.csv")
    threshold_output["sweep_data"].to_csv(csv_thresh_path, index=False)

    # 3. Export Comprehensive JSON Payload
    json_payload = {
        "timestamp": datetime.now().isoformat(),
        "champion_tuned_model": tuning_output["best_tuned_model_name"],
        "tuning_summary": table_rows,
        "operating_points": threshold_output["operating_points"],
        "top_features": diagnostics_output["feature_importances"][:15],
        "calibration_brier_scores": {
            k: v["brier_score"] for k, v in diagnostics_output["calibration"].items()
        },
    }
    json_path = os.path.join(target_results, "tuning_and_diagnostics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=4)

    # 4. Serialize Tuned Models to models/
    file_map = {
        "Support Vector Machine": "tuned_svm.joblib",
        "Random Forest": "tuned_random_forest.joblib",
        "K-Nearest Neighbors": "tuned_knn.joblib",
        "Logistic Regression": "tuned_logistic_regression.joblib",
    }
    saved_models = {}
    for name, model in tuning_output["fitted_tuned_models"].items():
        fname = file_map.get(name, f"tuned_{name.lower().replace(' ', '_')}.joblib")
        mpath = os.path.join(target_models, fname)
        joblib.dump(model, mpath)
        saved_models[name] = mpath

    best_tuned_path = os.path.join(target_models, "best_tuned_model.joblib")
    joblib.dump(tuning_output["fitted_tuned_models"][tuning_output["best_tuned_model_name"]], best_tuned_path)
    saved_models["best_tuned_model"] = best_tuned_path

    print(f"\n[*] Persisted Tuning & Diagnostic Artifacts:")
    print(f"    - Tuning Metrics CSV     : {os.path.basename(csv_tuning_path)}")
    print(f"    - Threshold Analysis CSV : {os.path.basename(csv_thresh_path)}")
    print(f"    - Diagnostics JSON       : {os.path.basename(json_path)}")
    print(f"    - Serialized Models      : {len(saved_models)} models saved in models/")

    return {
        "metrics_csv": csv_tuning_path,
        "threshold_csv": csv_thresh_path,
        "json": json_path,
        "models": saved_models,
    }


class SonarModelTuner:
    """
    Modular Day 6 Pipeline Class for Hyperparameter Tuning, Decision
    Threshold Optimization, and Model Diagnostics.
    """

    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state
        self.tuning_output = None
        self.threshold_output = None
        self.diagnostics_output = None
        self.saved_plots = None
        self.saved_artifacts = None

    def execute_tuning_and_diagnostics(self, X_train, y_train, X_test, y_test, baseline_results=None):
        self.tuning_output = tune_all_models(
            X_train,
            y_train,
            X_test,
            y_test,
            baseline_results=baseline_results,
            cv=self.cv,
            random_state=self.random_state,
        )

        champion_name = self.tuning_output["best_tuned_model_name"]
        champion_model = self.tuning_output["fitted_tuned_models"][champion_name]

        self.threshold_output = optimize_decision_thresholds(
            model=champion_model,
            X_test=X_test,
            y_test=y_test,
            model_name=champion_name,
        )

        self.diagnostics_output = compute_model_diagnostics(
            fitted_models=self.tuning_output["fitted_tuned_models"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cv=self.cv,
            random_state=self.random_state,
        )

        return {
            "tuning": self.tuning_output,
            "threshold": self.threshold_output,
            "diagnostics": self.diagnostics_output,
        }

    def save_all_artifacts(self, results_dir=None, models_dir=None):
        if self.tuning_output is None:
            raise ValueError("[ERROR] Execute execute_tuning_and_diagnostics() before saving artifacts.")

        self.saved_plots = generate_and_save_diagnostic_plots(
            tuning_output=self.tuning_output,
            threshold_output=self.threshold_output,
            diagnostics_output=self.diagnostics_output,
            output_dir=results_dir,
        )

        self.saved_artifacts = save_tuning_artifacts(
            tuning_output=self.tuning_output,
            threshold_output=self.threshold_output,
            diagnostics_output=self.diagnostics_output,
            results_dir=results_dir,
            models_dir=models_dir,
        )

        return {
            "plots": self.saved_plots,
            "artifacts": self.saved_artifacts,
        }


def run_full_tuning(
    processed_data_dir=None,
    models_dir=None,
    results_dir=None,
    cv=5,
    random_state=42,
):
    """
    Execute complete Day 6 pipeline:
      1. Load preprocessed splits from data/processed/
      2. Load Day 5 baseline metrics for comparison
      3. Run GridSearchCV hyperparameter optimization for all models
      4. Execute naval threat decision threshold calibration
      5. Compute learning curves, feature importances, and calibration curves
      6. Render 5 diagnostic figures to results/
      7. Persist tuned models to models/ and metrics to results/
    """
    from src.preprocessor import load_preprocessed_artifacts

    X_train, X_test, y_train, y_test, scaler, metadata = load_preprocessed_artifacts(
        processed_dir=processed_data_dir,
        models_dir=models_dir,
    )

    # Read Day 5 baseline metrics if available
    baseline_json_path = os.path.join(
        results_dir if results_dir is not None else DEFAULT_RESULTS_DIR,
        "model_evaluation_metrics.json",
    )
    baseline_results = None
    if os.path.exists(baseline_json_path):
        try:
            with open(baseline_json_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
                baseline_results = baseline_data.get("detailed_results", {})
        except Exception:
            baseline_results = None

    tuner = SonarModelTuner(cv=cv, random_state=random_state)
    results = tuner.execute_tuning_and_diagnostics(
        X_train,
        y_train,
        X_test,
        y_test,
        baseline_results=baseline_results,
    )
    saved = tuner.save_all_artifacts(results_dir=results_dir, models_dir=models_dir)

    print("\n" + "=" * 76)
    print("   DAY 6 HYPERPARAMETER TUNING & DIAGNOSTICS COMPLETED SUCCESSFULLY!")
    print("=" * 76)

    return {
        "tuner": tuner,
        "results": results,
        "saved": saved,
    }
