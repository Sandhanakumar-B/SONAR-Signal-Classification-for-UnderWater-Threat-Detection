"""
src/model_trainer.py
--------------------
Supervised model training, cross-validation, evaluation, and artifact
persistence module for the Sonar Signal Classification System (Day 5).

Supported Algorithms:
  1. Logistic Regression (Linear baseline with L2 regularization)
  2. K-Nearest Neighbors (KNN - Non-parametric distance-based classifier)
  3. Support Vector Machine (SVM - Maximum margin classifier with RBF kernel)
  4. Random Forest (Ensemble bagging decision forest classifier)

Pipeline Features:
  - Stratified K-Fold Cross-Validation on standardized training set (leakage-free)
  - Comprehensive Test Set Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Threat detection emphasis (Mine detection recall & false negative audit)
  - Publication-quality visualization generation (comparison bar charts, confusion matrices, ROC curves)
  - Serialization of all trained models and identification of the champion model
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# Canonical directories
DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)
DEFAULT_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def get_default_models(random_state=42):
    """
    Instantiate standard supervised classification models for sonar signal benchmark.

    Parameters
    ----------
    random_state : int, default=42
        Seed for reproducibility across stochastic models.

    Returns
    -------
    dict of str: estimator
        Dictionary mapping model names to instantiated scikit-learn estimators.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2,
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
        ),
    }
    return models


def evaluate_model_cv(model, X_train, y_train, cv=5, random_state=42):
    """
    Perform Stratified K-Fold Cross-Validation on the training partition.

    Parameters
    ----------
    model : estimator
        Scikit-learn classifier.
    X_train : pd.DataFrame or np.ndarray
        Standardized training features.
    y_train : pd.Series or np.ndarray
        Encoded binary labels.
    cv : int, default=5
        Number of stratified folds.
    random_state : int, default=42
        Random seed for fold splitting.

    Returns
    -------
    dict
        Cross-validation metrics with mean and standard deviation.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=False,
    )

    cv_summary = {
        "cv_folds": cv,
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "cv_precision_mean": float(np.mean(scores["test_precision"])),
        "cv_precision_std": float(np.std(scores["test_precision"])),
        "cv_recall_mean": float(np.mean(scores["test_recall"])),
        "cv_recall_std": float(np.std(scores["test_recall"])),
        "cv_f1_mean": float(np.mean(scores["test_f1"])),
        "cv_f1_std": float(np.std(scores["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
    }
    return cv_summary


def evaluate_model_test(model, X_test, y_test):
    """
    Evaluate fitted model performance on the held-out test partition.

    Parameters
    ----------
    model : estimator
        Fitted scikit-learn classifier.
    X_test : pd.DataFrame or np.ndarray
        Standardized test features.
    y_test : pd.Series or np.ndarray
        Encoded true test labels.

    Returns
    -------
    dict
        Test performance metrics, confusion matrix, probabilities, and classification report.
    """
    y_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
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

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=["Rock (0)", "Mine (1)"],
        output_dict=True,
    )

    return {
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "classification_report": clf_report,
    }


def train_and_evaluate_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models=None,
    cv=5,
    random_state=42,
):
    """
    Train and evaluate multiple supervised models with both Stratified CV and Test set evaluation.

    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Standardized training features.
    y_train : pd.Series or np.ndarray
        Encoded training labels.
    X_test : pd.DataFrame or np.ndarray
        Standardized test features.
    y_test : pd.Series or np.ndarray
        Encoded test labels.
    models : dict, optional
        Dictionary of models to train. Defaults to get_default_models().
    cv : int, default=5
        Folds for Stratified K-Fold CV.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    dict
        Comprehensive results containing fitted models, CV metrics, and test metrics.
    """
    if models is None:
        models = get_default_models(random_state=random_state)

    results = {}
    fitted_models = {}

    print("\n" + "=" * 76)
    print("   DAY 5: SUPERVISED MODEL TRAINING & CROSS-VALIDATION PIPELINE")
    print("=" * 76)
    print(f"[*] Training Samples: {len(X_train)} | Test Samples: {len(X_test)} | Features: {X_train.shape[1]}")
    print(f"[*] Cross-Validation : {cv}-Fold Stratified K-Fold (Random Seed: {random_state})\n")

    for name, model in models.items():
        print(f"--> Training & Evaluating: {name}...")

        # 1. Stratified K-Fold Cross-Validation on training data
        cv_metrics = evaluate_model_cv(
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            random_state=random_state,
        )

        # 2. Fit model on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # 3. Evaluate on held-out test set
        test_metrics = evaluate_model_test(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        # Combine metrics
        results[name] = {
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
        }

        print(
            f"    [CV  Scores] Acc: {cv_metrics['cv_accuracy_mean']:.4f} (+/- {cv_metrics['cv_accuracy_std']:.4f}) | "
            f"F1: {cv_metrics['cv_f1_mean']:.4f} | ROC-AUC: {cv_metrics['cv_roc_auc_mean']:.4f}"
        )
        print(
            f"    [Test Scores] Acc: {test_metrics['test_accuracy']:.4f} | "
            f"Prec: {test_metrics['test_precision']:.4f} | "
            f"Rec: {test_metrics['test_recall']:.4f} | "
            f"F1: {test_metrics['test_f1']:.4f} | "
            f"ROC-AUC: {test_metrics['test_roc_auc']:.4f} (TP={test_metrics['tp']}, FN={test_metrics['fn']})"
        )

    # Determine Champion / Best Model based on Test F1-Score (tie-breaker: CV F1)
    best_model_name = max(
        results.keys(),
        key=lambda k: (
            results[k]["test_metrics"]["test_f1"],
            results[k]["cv_metrics"]["cv_f1_mean"],
        ),
    )

    print("\n" + "-" * 76)
    print(f"[*] CHAMPION MODEL SELECTED: {best_model_name}")
    print(
        f"    Test Accuracy: {results[best_model_name]['test_metrics']['test_accuracy'] * 100:.2f}% | "
        f"Test F1-Score: {results[best_model_name]['test_metrics']['test_f1']:.4f} | "
        f"Threat Recall: {results[best_model_name]['test_metrics']['test_recall'] * 100:.2f}%"
    )
    print("-" * 76)

    return {
        "results": results,
        "fitted_models": fitted_models,
        "best_model_name": best_model_name,
    }


def generate_and_save_model_plots(results_dict, output_dir=None):
    """
    Generate publication-ready evaluation plots:
      1. Model Performance Comparison Bar Chart (CV & Test Metrics)
      2. Confusion Matrices for all models (highlighting Mine detection)
      3. Receiver Operating Characteristic (ROC) Curves

    Parameters
    ----------
    results_dict : dict
        Output dictionary from train_and_evaluate_all_models().
    output_dir : str, optional
        Target directory to save plots. Defaults to results/.

    Returns
    -------
    dict
        Paths to saved plot images.
    """
    target_dir = output_dir if output_dir is not None else DEFAULT_RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    results = results_dict["results"]
    model_names = list(results.keys())

    saved_plots = {}

    # Set aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    # =========================================================================
    # PLOT 1: Performance Comparison Bar Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(model_names))
    width = 0.15

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, metric in enumerate(["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]):
        vals = [results[m]["test_metrics"][metric] for m in model_names]
        offset = (i - 2) * width
        rects = ax.bar(x + offset, vals, width, label=metrics_names[i], color=palette[i], alpha=0.9, edgecolor="black", linewidth=0.6)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_title("Underwater Sonar Classification - Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", framealpha=0.95, ncol=5)
    plt.tight_layout()

    comp_path = os.path.join(target_dir, "model_comparison.png")
    fig.savefig(comp_path)
    plt.close(fig)
    saved_plots["model_comparison"] = comp_path

    # =========================================================================
    # PLOT 2: Confusion Matrices Grid
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        ax_cm = axes[i]
        cm = np.array(results[name]["test_metrics"]["confusion_matrix"])

        # Format annotations with percentages
        cm_sum = np.sum(cm)
        annot = np.array([
            [f"{cm[0, 0]}\n(TN: {cm[0, 0]/cm_sum:.1%})", f"{cm[0, 1]}\n(FP: {cm[0, 1]/cm_sum:.1%})"],
            [f"{cm[1, 0]}\n(FN: {cm[1, 0]/cm_sum:.1%})", f"{cm[1, 1]}\n(TP: {cm[1, 1]/cm_sum:.1%})"],
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            linewidths=1,
            linecolor="gray",
            annot_kws={"size": 11, "fontweight": "bold"},
        )
        acc_val = results[name]["test_metrics"]["test_accuracy"] * 100
        f1_val = results[name]["test_metrics"]["test_f1"]
        ax_cm.set_title(f"{name}\nAccuracy: {acc_val:.1f}% | F1: {f1_val:.3f}", fontsize=11, fontweight="bold")
        ax_cm.set_xlabel("Predicted Label", fontsize=10, fontweight="bold")
        ax_cm.set_ylabel("True Label", fontsize=10, fontweight="bold")
        ax_cm.set_xticklabels(["Rock (0)", "Mine (1)"], fontsize=10)
        ax_cm.set_yticklabels(["Rock (0)", "Mine (1)"], fontsize=10, rotation=0)

    plt.suptitle("Confusion Matrices on Test Set (Mine Threat vs Rock)", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    cm_path = os.path.join(target_dir, "confusion_matrices.png")
    fig.savefig(cm_path)
    plt.close(fig)
    saved_plots["confusion_matrices"] = cm_path

    # =========================================================================
    # PLOT 3: Receiver Operating Characteristic (ROC) Curves
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    roc_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, name in enumerate(model_names):
        fpr = results[name]["test_metrics"]["fpr"]
        tpr = results[name]["test_metrics"]["tpr"]
        auc = results[name]["test_metrics"]["test_roc_auc"]
        ax.plot(fpr, tpr, color=roc_colors[i], lw=2.2, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curves for Sonar Threat Classification (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)
    plt.tight_layout()

    roc_path = os.path.join(target_dir, "roc_curves.png")
    fig.savefig(roc_path)
    plt.close(fig)
    saved_plots["roc_curves"] = roc_path

    print(f"\n[*] Generated & Saved 3 Model Evaluation Visualizations to '{target_dir}':")
    for k, p in saved_plots.items():
        print(f"    - {k:<20} -> {os.path.relpath(p, os.path.join(target_dir, '..'))}")

    return saved_plots


def save_evaluation_metrics(results_dict, output_dir=None):
    """
    Export detailed metrics to both CSV and JSON formats in results/.

    Parameters
    ----------
    results_dict : dict
        Output dictionary from train_and_evaluate_all_models().
    output_dir : str, optional
        Target directory to save metric tables. Defaults to results/.

    Returns
    -------
    dict
        Paths to saved JSON and CSV files.
    """
    target_dir = output_dir if output_dir is not None else DEFAULT_RESULTS_DIR
    os.makedirs(target_dir, exist_ok=True)

    results = results_dict["results"]
    best_model_name = results_dict["best_model_name"]

    table_rows = []
    for name, data in results.items():
        cv = data["cv_metrics"]
        test = data["test_metrics"]
        table_rows.append({
            "Model": name,
            "CV_Accuracy_Mean": round(cv["cv_accuracy_mean"], 4),
            "CV_Accuracy_Std": round(cv["cv_accuracy_std"], 4),
            "CV_F1_Mean": round(cv["cv_f1_mean"], 4),
            "CV_ROC_AUC_Mean": round(cv["cv_roc_auc_mean"], 4),
            "Test_Accuracy": round(test["test_accuracy"], 4),
            "Test_Precision": round(test["test_precision"], 4),
            "Test_Recall": round(test["test_recall"], 4),
            "Test_F1_Score": round(test["test_f1"], 4),
            "Test_ROC_AUC": round(test["test_roc_auc"], 4),
            "True_Positives (TP)": test["tp"],
            "False_Negatives (FN)": test["fn"],
            "True_Negatives (TN)": test["tn"],
            "False_Positives (FP)": test["fp"],
            "Is_Champion": (name == best_model_name),
        })

    metrics_df = pd.DataFrame(table_rows)
    csv_path = os.path.join(target_dir, "model_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    json_payload = {
        "timestamp": datetime.now().isoformat(),
        "champion_model": best_model_name,
        "models_evaluated": list(results.keys()),
        "summary_table": table_rows,
        "detailed_results": results,
    }
    json_path = os.path.join(target_dir, "model_evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=4)

    print(f"\n[*] Exported Evaluation Metrics:")
    print(f"    - CSV Table : {os.path.relpath(csv_path, os.path.join(target_dir, '..'))}")
    print(f"    - JSON File : {os.path.relpath(json_path, os.path.join(target_dir, '..'))}")

    return {"csv": csv_path, "json": json_path, "dataframe": metrics_df}


def save_trained_models(fitted_models, best_model_name, output_dir=None):
    """
    Serialize all fitted scikit-learn models to models/ using Joblib,
    and persist a dedicated copy of the champion model as 'best_model.joblib'.

    Parameters
    ----------
    fitted_models : dict
        Dictionary of model_name -> fitted estimator.
    best_model_name : str
        Name of the best-performing model.
    output_dir : str, optional
        Target directory to save models. Defaults to models/.

    Returns
    -------
    dict
        Paths to serialized model artifacts.
    """
    target_dir = output_dir if output_dir is not None else DEFAULT_MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)

    saved_paths = {}

    file_name_mapping = {
        "Logistic Regression": "logistic_regression.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Support Vector Machine": "svm.joblib",
        "Random Forest": "random_forest.joblib",
    }

    for name, model in fitted_models.items():
        fname = file_name_mapping.get(name, f"{name.lower().replace(' ', '_')}.joblib")
        path = os.path.join(target_dir, fname)
        joblib.dump(model, path)
        saved_paths[name] = path

    # Save champion model alias
    best_path = os.path.join(target_dir, "best_model.joblib")
    joblib.dump(fitted_models[best_model_name], best_path)
    saved_paths["best_model"] = best_path

    print(f"\n[*] Serialized Trained Models into '{target_dir}':")
    for k, p in saved_paths.items():
        print(f"    - {k:<24} -> {os.path.basename(p)}")

    return saved_paths


class SonarModelTrainer:
    """
    Modular Trainer Pipeline Class for Sonar Signal Classification.

    Encapsulates dataset loading, model training, cross-validation, evaluation,
    plotting, and artifact persistence.
    """

    def __init__(self, cv=5, random_state=42):
        self.cv = cv
        self.random_state = random_state
        self.models = get_default_models(random_state=random_state)
        self.training_results = None
        self.saved_plots = None
        self.saved_metrics = None
        self.saved_models = None

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        self.training_results = train_and_evaluate_all_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=self.models,
            cv=self.cv,
            random_state=self.random_state,
        )
        return self.training_results

    def save_artifacts(self, results_dir=None, models_dir=None):
        if self.training_results is None:
            raise ValueError("[ERROR] Call fit_and_evaluate() before saving artifacts.")

        self.saved_plots = generate_and_save_model_plots(
            self.training_results,
            output_dir=results_dir,
        )
        self.saved_metrics = save_evaluation_metrics(
            self.training_results,
            output_dir=results_dir,
        )
        self.saved_models = save_trained_models(
            self.training_results["fitted_models"],
            self.training_results["best_model_name"],
            output_dir=models_dir,
        )
        return {
            "plots": self.saved_plots,
            "metrics": self.saved_metrics,
            "models": self.saved_models,
        }


def run_full_training(
    processed_data_dir=None,
    models_dir=None,
    results_dir=None,
    cv=5,
    random_state=42,
):
    """
    Execute the end-to-end Day 5 model training and evaluation pipeline:
      1. Loads preprocessed artifacts from data/processed/
      2. Performs 5-Fold Stratified Cross-Validation on training data
      3. Fits all 4 supervised models on training set
      4. Evaluates performance on unseen test set
      5. Identifies champion model
      6. Generates 3 visualization figures in results/
      7. Exports metrics tables (CSV & JSON) in results/
      8. Persists serialized models and best_model.joblib in models/

    Returns
    -------
    dict
        Execution summary and paths.
    """
    # Import loader from preprocessor module
    from src.preprocessor import load_preprocessed_artifacts

    X_train, X_test, y_train, y_test, scaler, metadata = load_preprocessed_artifacts(
        processed_dir=processed_data_dir,
        models_dir=models_dir,
    )

    trainer = SonarModelTrainer(cv=cv, random_state=random_state)
    results = trainer.fit_and_evaluate(X_train, y_train, X_test, y_test)
    artifacts = trainer.save_artifacts(results_dir=results_dir, models_dir=models_dir)

    print("\n" + "=" * 76)
    print("   DAY 5 MODEL TRAINING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 76)

    return {
        "trainer": trainer,
        "results": results,
        "artifacts": artifacts,
    }
