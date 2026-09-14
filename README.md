# Machine Learning-Based Sonar Signal Classification System for Underwater Threat Detection

An academic Artificial Intelligence and Machine Learning (AI/ML) project designed to classify underwater sonar signals into **Mines (Threats)** or **Rocks (Non-threats)** using supervised machine learning algorithms.

---

## 📌 Project Overview

Naval defense systems rely heavily on acoustic sonar echoes to monitor underwater environments. Distinguishing between explosive underwater naval mines and benign geological rock formations is a critical challenge. 

This project implements an end-to-end Machine Learning pipeline that:
1. Loads and explores raw acoustic sonar frequency data (UCI Sonar Dataset).
2. Cleans, preprocesses, and standardizes continuous signal features.
3. Trains and evaluates multiple supervised machine learning classification models.
4. Compares performance metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
5. Provides an interactive prediction interface for classifying new sonar signals.

---

## 📂 Project Directory Structure

```text
sonar-threat-detection/
│
├── data/
│   ├── .gitkeep
│   ├── sonar.csv                  # Raw UCI Sonar dataset (208 instances x 60 features + 1 label)
│   └── processed/                 # Day 4 preprocessed & scaled datasets
│       ├── X_train.csv            # Standardized training features (166 x 60)
│       ├── X_test.csv             # Standardized testing features (42 x 60)
│       ├── y_train.csv            # Encoded training labels (166 x 1: M->1, R->0)
│       ├── y_test.csv             # Encoded testing labels (42 x 1: M->1, R->0)
│       └── preprocessing_metadata.json # Split ratios, scaling stats, and class balance log
├── notebooks/                     # Jupyter notebooks for interactive EDA and experiments
├── src/                           # Modular Python source code
│   ├── __init__.py
│   ├── data_loader.py             # Data ingestion and preliminary exploration module
│   ├── eda.py                     # Exploratory data analysis and visualization pipeline
│   ├── preprocessor.py            # Day 4 preprocessing, feature scaling & splitting pipeline
│   ├── model_trainer.py           # Day 5 model training, cross-validation & evaluation module
│   └── tuner.py                   # Day 6 hyperparameter tuning, threshold optimization & diagnostics
├── models/                        # Serialized models and transformers
│   ├── .gitkeep
│   ├── scaler.joblib              # Fitted StandardScaler (leakage-free, fit strictly on train)
│   ├── logistic_regression.joblib # Day 5 Logistic Regression baseline
│   ├── knn.joblib                 # Day 5 KNN baseline
│   ├── svm.joblib                 # Day 5 Support Vector Machine baseline
│   ├── random_forest.joblib       # Day 5 Random Forest baseline
│   ├── best_model.joblib          # Day 5 Champion baseline model (SVM)
│   ├── tuned_svm.joblib           # Day 6 Tuned SVM model
│   ├── tuned_random_forest.joblib # Day 6 Tuned Random Forest model
│   ├── tuned_knn.joblib           # Day 6 Tuned KNN model
│   ├── tuned_logistic_regression.joblib # Day 6 Tuned Logistic Regression model
│   └── best_tuned_model.joblib    # Day 6 Champion Tuned Model (SVM with 92.86% Acc, 100% Threat Recall)
├── results/                       # Evaluation metrics, confusion matrices, and saved plots
│   ├── class_distribution.png     # Class distribution bar chart (Mines vs Rocks)
│   ├── mean_spectral_signature.png# Mean energy spectral profile across 60 frequency bands
│   ├── feature_distributions.png  # Histograms and KDEs for representative frequency bands
│   ├── feature_boxplots.png       # Boxplots comparing class separations on key features
│   ├── correlation_heatmap.png    # Correlation matrix heatmap of acoustic frequencies
│   ├── model_comparison.png       # Day 5 performance comparison bar chart across models
│   ├── confusion_matrices.png     # Day 5 2x2 confusion matrix grid on test set
│   ├── roc_curves.png             # Day 5 ROC curves and AUC comparison
│   ├── model_metrics.csv          # Day 5 comparative metrics spreadsheet
│   ├── model_evaluation_metrics.json # Day 5 structured metrics and classification reports
│   ├── tuning_comparison.png      # Day 6 baseline vs tuned F1-score comparison
│   ├── threshold_optimization.png # Day 6 precision-recall-cost vs threshold trade-off curves
│   ├── learning_curves.png        # Day 6 bias-variance learning curve trajectories
│   ├── feature_importance.png     # Day 6 top 15 most discriminative sonar frequency bands
│   ├── calibration_curves.png     # Day 6 probability reliability diagrams & Brier scores
│   ├── tuning_metrics.csv         # Day 6 hyperparameter tuning metrics table
│   ├── threshold_analysis.csv     # Day 6 decision threshold sweep analysis table
│   └── tuning_and_diagnostics.json# Day 6 structured diagnostics and operating point configurations
├── .gitignore                     # Files and folders to exclude from version control
├── README.md                      # Project documentation and daily tracking
├── requirements.txt               # Project dependencies and libraries
├── test_environment.py            # Environment and dependency verification script
├── load_data.py                   # Day 2 dataset loading execution script
├── eda.py                         # Day 3 exploratory data analysis execution script
├── preprocess.py                  # Day 4 preprocessing and feature scaling execution script
├── train.py                       # Day 5 model training and cross-validation execution script
└── tune.py                        # Day 6 hyperparameter tuning and diagnostics execution script
```

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+
- **Numerical Computing:** NumPy
- **Data Manipulation:** Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-Learn
- **Model / Transformer Persistence:** Joblib
- **Version Control:** Git & GitHub

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd "SONAR Signal Classification"
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify the Environment (Day 1)
Run the environment verification test script:
```bash
python test_environment.py
```

### 5. Load and Inspect Dataset (Day 2)
Run the dataset loading script:
```bash
python load_data.py
# or run the modular component directly:
python src/data_loader.py
```

### 6. Exploratory Data Analysis & Visualizations (Day 3)
Run the comprehensive EDA workflow and generate plots:
```bash
python eda.py
```
Generated plots will be saved into the [`results/`](results/) folder.

### 7. Data Preprocessing, Feature Scaling & Train-Test Splitting (Day 4)
Run the preprocessing and feature scaling pipeline:
```bash
python preprocess.py
```
Preprocessed datasets are saved to [`data/processed/`](data/processed/) and the fitted scaler is saved to [`models/scaler.joblib`](models/scaler.joblib).

### 8. Supervised Model Training & Cross-Validation (Day 5)
Run the supervised model training, cross-validation, and evaluation pipeline:
```bash
python train.py
```

#### Day 5 Benchmark Performance Summary (Test Set):
| Model | 5-Fold CV Acc (± Std) | Test Accuracy | Precision | Threat Recall | F1-Score | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Support Vector Machine (SVM)** | **80.77% (± 7.89%)** | **92.86%** | **88.00%** | **100.00%** | **0.9362** | **0.9727** | **Champion 🏆** |
| **Logistic Regression** | 78.95% (± 4.06%) | 83.33% | 82.61% | 86.36% | 0.8444 | 0.9045 | Baseline |
| **Random Forest** | 76.54% (± 6.00%) | 80.95% | 79.17% | 86.36% | 0.8261 | 0.9386 | Ensemble |
| **K-Nearest Neighbors (KNN)** | 76.47% (± 5.98%) | 78.57% | 76.00% | 86.36% | 0.8085 | 0.9500 | Non-parametric |

All trained models are saved to [`models/`](models/) and comparative visualization figures (`model_comparison.png`, `confusion_matrices.png`, `roc_curves.png`) are saved to [`results/`](results/).

### 9. Hyperparameter Tuning, Threshold Optimization & Model Diagnostics (Day 6)
Run the automated grid-search hyperparameter tuning, naval threat threshold calibration, and deep diagnostic pipeline:
```bash
python tune.py
```

#### Day 6 Tuned Performance Benchmark Summary (Test Set):
| Model | Optimal Hyperparameters | 5-Fold CV F1 | Test Accuracy | Threat Recall | F1-Score | ROC-AUC | Delta F1 vs Day 5 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Support Vector Machine (SVM)** | `C=1.0`, `gamma='scale'`, `kernel='rbf'` | **0.8312** | **92.86%** | **100.00%** | **0.9362** | **0.9727** | **+0.0000** | **Champion 🏆** |
| **Random Forest** | `n_estimators=200`, `max_depth=None`, `max_features='log2'`, `min_samples_leaf=2` | 0.8324 | 85.71% | 95.45% | 0.8750 | 0.9227 | **+0.0489** | Tuned Ensemble |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=4`, `weights='distance'`, `metric='manhattan'` | **0.8509** | 85.71% | 95.45% | 0.8750 | 0.9705 | **+0.0665** | Tuned Distance |
| **Logistic Regression** | `C=0.05`, `solver='lbfgs'` | 0.8420 | 78.57% | 77.27% | 0.7907 | 0.8841 | -0.0537 | Regularized Baseline |

#### 🎯 Naval Threat Decision Threshold Optimization (SVM):
Underwater naval mine detection demands near-zero False Negatives (FN = lethal missed mine). Decision thresholds ($\tau$) were swept over $[0.01, 0.99]$:
- **Default Operating Point ($\tau = 0.50$):** Accuracy = 88.1%, Recall = 90.9%, F1 = 0.8889, FN = 2, Threat Cost = 13.0
- **Optimized & Cost-Safe Operating Point ($\tau^* = 0.43$):** Accuracy = **92.9%**, Precision = **88.0%**, Recall = **100.0%**, F1 = **0.9362**, **FN = 0 (Zero Mines Missed)**, Threat Cost = **3.0**

#### 🔬 Key Diagnostic Insights:
1. **Learning Curves (`results/learning_curves.png`):** Low bias and converging training/CV curves confirm no high-variance overfitting.
2. **Frequency Importance (`results/feature_importance.png`):** Frequency bands `Freq_11`, `Freq_12`, `Freq_09`, `Freq_10`, and `Freq_36` exhibit the highest discrimination power.
3. **Probability Calibration (`results/calibration_curves.png`):** SVM demonstrated superior probability reliability with a low Brier Score of **0.0735**.

---

## 📅 Daily Progress Tracker

| Day | Milestone / Task | Status |
| :--- | :--- | :--- |
| **Day 1** | Project setup, folder architecture, dependency configuration, and environment verification | Completed ✅ |
| **Day 2** | Dataset acquisition & preliminary data loading (Pandas loading, inspection, class distribution) | Completed ✅ |
| **Day 3** | Exploratory Data Analysis (EDA), Statistical Analysis & Visualizations | Completed ✅ |
| **Day 4** | Data Preprocessing, Feature Scaling & Train-Test Splitting (Stratified 80/20, StandardScaler, Data Leakage Prevention) | Completed ✅ |
| **Day 5** | Supervised Model Training & Cross-Validation (Logistic Regression, KNN, SVM, Random Forest) | Completed ✅ |
| **Day 6** | Hyperparameter Tuning, Threshold Optimization & Comprehensive Model Diagnostics | Completed ✅ |

---

## 👥 Author & Academic Details
- **Project Title:** Machine Learning-Based Sonar Signal Classification System for Underwater Threat Detection
- **Domain:** Artificial Intelligence & Machine Learning (AI/ML)
