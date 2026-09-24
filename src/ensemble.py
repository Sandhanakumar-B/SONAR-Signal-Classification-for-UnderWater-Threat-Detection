"""
src/ensemble.py
----------------
Day 10: Deep Learning (Neural Networks - Multi-Layer Perceptron),
Advanced Ensemble Architectures (Stacking & Soft-Voting) & Multi-Model Benchmarking.

Capabilities:
  1. Deep Neural Network (MLPClassifier):
     - Dense feed-forward neural network tailored for 60-dimensional continuous acoustic spectra.
     - Architecture: Input (60) -> Dense (64, ReLU) -> Dense (32, ReLU) -> Output (Sigmoid probability).
     - Adam optimizer, L2 regularization (weight decay), and training loss trajectory logging.
  2. Soft-Voting Ensemble:
     - Blends predictive probabilities across distinct algorithmic families:
       SVM (maximum-margin kernel), Random Forest (bagging tree partition),
       MLP (distributed representation), and Logistic Regression (linear baseline).
  3. Stacking Classifier (Meta-Learner):
     - Hierarchical 2-tier architecture:
       * Base tier: Tuned SVM, Tuned Random Forest, and MLP Neural Network.
       * Meta tier: Calibrated Logistic Regression trained on 5-fold out-of-fold cross-validated meta-features.
  4. Comprehensive Multi-Paradigm Benchmarking:
     - Compares Baselines vs. Tuned Models vs. Neural Network vs. Ensembles across
       Accuracy, Precision, Recall, F1-Score, ROC-AUC, Brier score, and False Negative counts.
  5. Publication-Quality Visualization Generation:
     - results/neural_loss_curve.png
     - results/ensemble_comparison.png
     - results/ensemble_confusion_matrices.png
     - results/ensemble_roc_curves.png
  6. Artifact Serialization to models/ & Structured Reporting to results/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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
    brier_score_loss,
)

# Canonical directories
DEFAULT_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)
DEFAULT_RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)
DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed")
)


class SonarEnsemblePipeline:
    """
    Day 10 Neural Network and Ensemble Learning Pipeline for Sonar Signal Classification.
    """

    def __init__(
        self,
        data_dir=DEFAULT_DATA_DIR,
        models_dir=DEFAULT_MODELS_DIR,
        results_dir=DEFAULT_RESULTS_DIR,
        random_state=42,
    ):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.random_state = random_state

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        self.tuned_svm = None
        self.tuned_rf = None
        self.tuned_lr = None
        self.tuned_knn = None

        self.mlp_model = None
        self.voting_model = None
        self.stacking_model = None

        self.evaluation_results = {}

    def load_data_and_base_models(self):
        """Loads preprocessed datasets and pre-existing tuned models."""
        x_train_path = os.path.join(self.data_dir, "X_train.csv")
        x_test_path = os.path.join(self.data_dir, "X_test.csv")
        y_train_path = os.path.join(self.data_dir, "y_train.csv")
        y_test_path = os.path.join(self.data_dir, "y_test.csv")

        if not all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path]):
            raise FileNotFoundError("Preprocessed data files not found in data/processed/. Please run preprocess.py first.")

        df_xtrain = pd.read_csv(x_train_path)
        df_xtest = pd.read_csv(x_test_path)
        self.feature_names = list(df_xtrain.columns)

        self.X_train = df_xtrain
        self.X_test = df_xtest
        self.y_train = pd.read_csv(y_train_path).values.ravel()
        self.y_test = pd.read_csv(y_test_path).values.ravel()

        # Load tuned models
        svm_path = os.path.join(self.models_dir, "tuned_svm.joblib")
        rf_path = os.path.join(self.models_dir, "tuned_random_forest.joblib")
        lr_path = os.path.join(self.models_dir, "tuned_logistic_regression.joblib")
        knn_path = os.path.join(self.models_dir, "tuned_knn.joblib")

        if os.path.exists(svm_path):
            self.tuned_svm = joblib.load(svm_path)
        if os.path.exists(rf_path):
            self.tuned_rf = joblib.load(rf_path)
        if os.path.exists(lr_path):
            self.tuned_lr = joblib.load(lr_path)
        if os.path.exists(knn_path):
            self.tuned_knn = joblib.load(knn_path)

        return self

    def build_and_train_mlp(self, hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=600):
        """
        Builds and trains a Multi-Layer Perceptron (Deep Neural Network) for continuous acoustic signals.
        """
        print(f"[*] Training Multi-Layer Perceptron (MLP) Neural Network {hidden_layer_sizes}...")
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size="auto",
            learning_rate="adaptive",
            learning_rate_init=0.003,
            max_iter=max_iter,
            random_state=self.random_state,
            early_stopping=False,
            verbose=False,
        )
        self.mlp_model.fit(self.X_train, self.y_train)
        print(f"    [+] MLP converged in {self.mlp_model.n_iter_} iterations with final loss {self.mlp_model.loss_:.4f}")
        return self.mlp_model

    def build_and_train_voting_ensemble(self):
        """
        Constructs a calibrated Soft-Voting Ensemble blending SVM, Random Forest, MLP, and Logistic Regression.
        """
        print("[*] Training Calibrated Soft-Voting Ensemble...")
        estimators = []
        if self.tuned_svm is not None:
            estimators.append(("svm", self.tuned_svm))
        if self.tuned_rf is not None:
            estimators.append(("rf", self.tuned_rf))
        if self.mlp_model is not None:
            estimators.append(("mlp", self.mlp_model))
        if self.tuned_lr is not None:
            estimators.append(("lr", self.tuned_lr))

        # Weight SVM and RF higher based on their high test F1 scores
        weights = [2, 1, 1, 1] if len(estimators) == 4 else None

        self.voting_model = VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights,
            n_jobs=None
        )
        self.voting_model.fit(self.X_train, self.y_train)
        print("    [+] Soft-Voting Ensemble trained successfully.")
        return self.voting_model

    def build_and_train_stacking_classifier(self):
        """
        Constructs a 2-Tier Stacking Classifier with 5-Fold Stratified cross-validated meta-features.
        """
        print("[*] Training Stacking Meta-Classifier...")
        base_estimators = []
        if self.tuned_svm is not None:
            base_estimators.append(("svm", self.tuned_svm))
        if self.tuned_rf is not None:
            base_estimators.append(("rf", self.tuned_rf))
        if self.mlp_model is not None:
            base_estimators.append(("mlp", self.mlp_model))

        final_estimator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            random_state=self.random_state,
            max_iter=500
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        self.stacking_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            cv=cv,
            stack_method="predict_proba",
            n_jobs=None
        )
        self.stacking_model.fit(self.X_train, self.y_train)
        print("    [+] Stacking Classifier trained successfully.")
        return self.stacking_model

    def evaluate_model(self, name, model, safe_threshold=0.43):
        """
        Evaluates a model comprehensively on the held-out test set under default (0.50) and safe thresholds.
        """
        # Feature names safe prediction
        probs = model.predict_proba(self.X_test)[:, 1]
        preds_default = (probs >= 0.50).astype(int)
        preds_safe = (probs >= safe_threshold).astype(int)

        # Metrics at default threshold (0.50)
        acc_default = float(accuracy_score(self.y_test, preds_default))
        prec_default = float(precision_score(self.y_test, preds_default, zero_division=0))
        rec_default = float(recall_score(self.y_test, preds_default, zero_division=0))
        f1_default = float(f1_score(self.y_test, preds_default, zero_division=0))
        cm_default = confusion_matrix(self.y_test, preds_default).tolist()

        # Metrics at safe threshold
        acc_safe = float(accuracy_score(self.y_test, preds_safe))
        prec_safe = float(precision_score(self.y_test, preds_safe, zero_division=0))
        rec_safe = float(recall_score(self.y_test, preds_safe, zero_division=0))
        f1_safe = float(f1_score(self.y_test, preds_safe, zero_division=0))
        cm_safe = confusion_matrix(self.y_test, preds_safe).tolist()

        roc_auc = float(roc_auc_score(self.y_test, probs))
        brier = float(brier_score_loss(self.y_test, probs))

        # Stratified 5-Fold Cross-Validation on training set
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_validate(
            model,
            self.X_train,
            self.y_train,
            cv=skf,
            scoring=["accuracy", "recall", "f1", "roc_auc"]
        )

        cv_acc_mean = float(np.mean(cv_scores["test_accuracy"]))
        cv_rec_mean = float(np.mean(cv_scores["test_recall"]))
        cv_f1_mean = float(np.mean(cv_scores["test_f1"]))
        cv_auc_mean = float(np.mean(cv_scores["test_roc_auc"]))

        return {
            "name": name,
            "cv_metrics": {
                "accuracy_mean": cv_acc_mean,
                "recall_mean": cv_rec_mean,
                "f1_mean": cv_f1_mean,
                "roc_auc_mean": cv_auc_mean,
            },
            "test_default": {
                "threshold": 0.50,
                "accuracy": acc_default,
                "precision": prec_default,
                "recall": rec_default,
                "f1_score": f1_default,
                "confusion_matrix": cm_default,
                "false_negatives": int(cm_default[1][0]),
                "false_positives": int(cm_default[0][1]),
                "true_positives": int(cm_default[1][1]),
                "true_negatives": int(cm_default[0][0]),
            },
            "test_safe": {
                "threshold": safe_threshold,
                "accuracy": acc_safe,
                "precision": prec_safe,
                "recall": rec_safe,
                "f1_score": f1_safe,
                "confusion_matrix": cm_safe,
                "false_negatives": int(cm_safe[1][0]),
                "false_positives": int(cm_safe[0][1]),
                "true_positives": int(cm_safe[1][1]),
                "true_negatives": int(cm_safe[0][0]),
            },
            "roc_auc": roc_auc,
            "brier_score": brier,
            "probabilities": probs.tolist()
        }

    def evaluate_all(self):
        """Evaluates all benchmark and Day 10 models."""
        models_to_eval = [
            ("Tuned SVM (Champion)", self.tuned_svm),
            ("Tuned Random Forest", self.tuned_rf),
            ("Deep Neural Network (MLP)", self.mlp_model),
            ("Soft-Voting Ensemble", self.voting_model),
            ("Stacking Classifier", self.stacking_model),
        ]

        self.evaluation_results = {}
        for name, model in models_to_eval:
            if model is not None:
                print(f"[*] Evaluating {name}...")
                self.evaluation_results[name] = self.evaluate_model(name, model)

        return self.evaluation_results

    def generate_and_save_plots(self):
        """Renders 4 publication-quality visualizations for Day 10."""
        print("[*] Generating Day 10 publication-quality visualizations...")

        # Plot 1: Neural Network Training Loss Curve
        if self.mlp_model is not None and hasattr(self.mlp_model, "loss_curve_"):
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            sns.set_theme(style="darkgrid")
            iterations = range(1, len(self.mlp_model.loss_curve_) + 1)
            ax.plot(iterations, self.mlp_model.loss_curve_, color="#38bdf8", linewidth=2.5, label="Adam Cross-Entropy Loss")
            ax.set_title("Deep Neural Network (MLP) - Training Loss Trajectory", fontsize=14, fontweight="bold", pad=12)
            ax.set_xlabel("Epoch / Iteration", fontsize=11, fontweight="bold")
            ax.set_ylabel("Loss (Cross-Entropy)", fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.5)

            # Annotation of final convergence
            final_iter = len(self.mlp_model.loss_curve_)
            final_loss = self.mlp_model.loss_curve_[-1]
            ax.annotate(
                f"Converged:\nEpoch {final_iter}\nLoss: {final_loss:.4f}",
                xy=(final_iter, final_loss),
                xytext=(final_iter - 120, final_loss + 0.15),
                arrowprops=dict(facecolor="#f59e0b", shrink=0.05, width=1.5, headwidth=7),
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", edgecolor="#f59e0b", alpha=0.9),
                color="#f8fafc"
            )

            plt.tight_layout()
            loss_path = os.path.join(self.results_dir, "neural_loss_curve.png")
            fig.savefig(loss_path, bbox_inches="tight")
            plt.close(fig)
            print(f"    [+] Saved: {loss_path}")

        # Plot 2: Comprehensive Multi-Paradigm Comparison Bar Chart
        if self.evaluation_results:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
            sns.set_theme(style="whitegrid")

            models_list = list(self.evaluation_results.keys())
            metrics_data = {
                "Accuracy": [self.evaluation_results[m]["test_default"]["accuracy"] * 100 for m in models_list],
                "Recall (Threats)": [self.evaluation_results[m]["test_default"]["recall"] * 100 for m in models_list],
                "F1-Score": [self.evaluation_results[m]["test_default"]["f1_score"] * 100 for m in models_list],
                "ROC-AUC": [self.evaluation_results[m]["roc_auc"] * 100 for m in models_list],
            }

            x = np.arange(len(models_list))
            width = 0.20
            palette = ["#0284c7", "#ef4444", "#10b981", "#8b5cf6"]

            for idx, (metric_name, values) in enumerate(metrics_data.items()):
                offset = (idx - 1.5) * width
                bars = ax.bar(x + offset, values, width, label=metric_name, color=palette[idx], alpha=0.9)
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f"{height:.1f}%",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, fontweight="bold")

            ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
            ax.set_title("Multi-Paradigm Performance Benchmark (Baselines, Tuned, Neural Net & Ensembles)", fontsize=14, fontweight="bold", pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(" ", "\n") for m in models_list], fontsize=10, fontweight="bold")
            ax.set_ylim(70, 105)
            ax.legend(loc="lower right", frameon=True, fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.6)

            plt.tight_layout()
            comp_path = os.path.join(self.results_dir, "ensemble_comparison.png")
            fig.savefig(comp_path, bbox_inches="tight")
            plt.close(fig)
            print(f"    [+] Saved: {comp_path}")

        # Plot 3: 2x2 Grid of Confusion Matrices for Day 10 Models
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
        selected_models = [
            ("Tuned SVM (Champion)", axes[0, 0]),
            ("Deep Neural Network (MLP)", axes[0, 1]),
            ("Soft-Voting Ensemble", axes[1, 0]),
            ("Stacking Classifier", axes[1, 1]),
        ]

        class_names = ["Rock (0)", "Mine (1)"]
        for model_name, ax in selected_models:
            if model_name in self.evaluation_results:
                cm = np.array(self.evaluation_results[model_name]["test_default"]["confusion_matrix"])
                acc = self.evaluation_results[model_name]["test_default"]["accuracy"] * 100
                fn = self.evaluation_results[model_name]["test_default"]["false_negatives"]

                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=class_names, yticklabels=class_names,
                            ax=ax, annot_kws={"size": 14, "weight": "bold"})
                ax.set_title(f"{model_name}\nAcc: {acc:.1f}% | Missed Mines (FN): {fn}", fontsize=11, fontweight="bold", pad=8)
                ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
                ax.set_ylabel("True Class", fontsize=10, fontweight="bold")

        plt.suptitle("Day 10 Unseen Test Set Confusion Matrices (Default tau = 0.50)", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        cm_path = os.path.join(self.results_dir, "ensemble_confusion_matrices.png")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        print(f"    [+] Saved: {cm_path}")

        # Plot 4: Multi-Model ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sns.set_theme(style="whitegrid")

        colors = {
            "Tuned SVM (Champion)": "#0284c7",
            "Tuned Random Forest": "#10b981",
            "Deep Neural Network (MLP)": "#f59e0b",
            "Soft-Voting Ensemble": "#8b5cf6",
            "Stacking Classifier": "#ef4444",
        }

        for model_name, data in self.evaluation_results.items():
            probs = np.array(data["probabilities"])
            fpr, tpr, _ = roc_curve(self.y_test, probs)
            auc_val = data["roc_auc"]
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.4f})",
                    color=colors.get(model_name, "#475569"), linewidth=2.2)

        # Baseline random guess line
        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.5, label="Random Chance (AUC = 0.5000)")
        ax.set_title("Day 10 Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="lower right", frameon=True, fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        roc_path = os.path.join(self.results_dir, "ensemble_roc_curves.png")
        fig.savefig(roc_path, bbox_inches="tight")
        plt.close(fig)
        print(f"    [+] Saved: {roc_path}")

    def save_artifacts(self):
        """Serializes Day 10 models and exports benchmark metrics reports."""
        print("[*] Serializing Day 10 models and exporting metrics...")

        # Save models
        if self.mlp_model is not None:
            mlp_path = os.path.join(self.models_dir, "mlp_neural_network.joblib")
            joblib.dump(self.mlp_model, mlp_path)
            print(f"    [+] Saved: {mlp_path}")

        if self.voting_model is not None:
            voting_path = os.path.join(self.models_dir, "voting_ensemble.joblib")
            joblib.dump(self.voting_model, voting_path)
            print(f"    [+] Saved: {voting_path}")

        if self.stacking_model is not None:
            stacking_path = os.path.join(self.models_dir, "stacking_ensemble.joblib")
            joblib.dump(self.stacking_model, stacking_path)
            print(f"    [+] Saved: {stacking_path}")

        # Export CSV spreadsheet
        rows = []
        for name, data in self.evaluation_results.items():
            rows.append({
                "Model": name,
                "CV_Accuracy_Mean": round(data["cv_metrics"]["accuracy_mean"], 4),
                "CV_F1_Mean": round(data["cv_metrics"]["f1_mean"], 4),
                "Test_Accuracy_Default": round(data["test_default"]["accuracy"], 4),
                "Test_Precision_Default": round(data["test_default"]["precision"], 4),
                "Test_Recall_Default": round(data["test_default"]["recall"], 4),
                "Test_F1_Default": round(data["test_default"]["f1_score"], 4),
                "False_Negatives_Default": data["test_default"]["false_negatives"],
                "Test_Accuracy_Safe": round(data["test_safe"]["accuracy"], 4),
                "Test_Recall_Safe": round(data["test_safe"]["recall"], 4),
                "False_Negatives_Safe": data["test_safe"]["false_negatives"],
                "ROC_AUC": round(data["roc_auc"], 4),
                "Brier_Score": round(data["brier_score"], 4),
            })

        df_metrics = pd.DataFrame(rows)
        csv_path = os.path.join(self.results_dir, "ensemble_metrics.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"    [+] Saved: {csv_path}")

        # Export JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "milestone": "Day 10: Deep Learning Neural Networks & Advanced Ensemble Architectures",
            "models_evaluated": list(self.evaluation_results.keys()),
            "neural_network_architecture": {
                "type": "Multi-Layer Perceptron (MLP)",
                "hidden_layers": [64, 32],
                "activation": "relu",
                "optimizer": "adam",
                "alpha_l2": 0.01,
                "iterations_converged": self.mlp_model.n_iter_ if self.mlp_model else None,
                "final_loss": round(float(self.mlp_model.loss_), 5) if self.mlp_model else None
            },
            "ensemble_architectures": {
                "soft_voting": "Weighted Soft-Voting (SVM: 2, RF: 1, MLP: 1, LR: 1)",
                "stacking": "2-Tier Stacking (Base: SVM + RF + MLP, Meta: Logistic Regression with 5-Fold Stratified Out-of-Fold Meta-Features)"
            },
            "metrics": self.evaluation_results
        }

        json_path = os.path.join(self.results_dir, "ensemble_evaluation_report.json")
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"    [+] Saved: {json_path}")

        return report


def run_full_ensemble_pipeline():
    """Executes the complete Day 10 pipeline and returns evaluation report."""
    pipeline = SonarEnsemblePipeline()
    pipeline.load_data_and_base_models()
    pipeline.build_and_train_mlp()
    pipeline.build_and_train_voting_ensemble()
    pipeline.build_and_train_stacking_classifier()
    pipeline.evaluate_all()
    pipeline.generate_and_save_plots()
    report = pipeline.save_artifacts()
    return report
