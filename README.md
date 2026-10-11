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
│   ├── tuner.py                   # Day 6 hyperparameter tuning, threshold optimization & diagnostics
│   ├── predictor.py               # Day 7 real-time inference and prediction engine (updated Day 9/10)
│   ├── explainer.py               # Day 8 model explainability, SHAP & error analysis module
│   └── ensemble.py                # Day 10 Deep Learning (MLP) & Stacking/Voting Ensemble module
├── templates/                     # Day 9 Flask HTML templates
│   └── index.html                 # Tactical Sonar Defense Dashboard (dark HUD UI with model selector)
├── static/                        # Day 9 web static assets
│   ├── css/
│   │   └── style.css              # Naval dark theme design system (glassmorphism, animations)
│   └── js/
│       └── app.js                 # Client-side state, API calls, canvas spectral renderer
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
│   ├── best_tuned_model.joblib    # Day 6 Champion Tuned Model (SVM with 92.86% Acc, 100% Threat Recall)
│   ├── mlp_neural_network.joblib  # Day 10 Deep Multi-Layer Perceptron (64-32 Dense) Neural Network
│   ├── voting_ensemble.joblib     # Day 10 Calibrated Soft-Voting Ensemble (SVM, RF, MLP, LR)
│   └── stacking_ensemble.joblib   # Day 10 2-Tier Stacking Meta-Classifier (Meta: Logistic Regression)
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
│   ├── tuning_and_diagnostics.json# Day 6 structured diagnostics and operating point configurations
│   ├── permutation_importance.png # Day 8 test-set permutation feature importance plot
│   ├── shap_summary.png           # Day 8 SHAP global feature attribution bar chart
│   ├── shap_beeswarm.png          # Day 8 SHAP beeswarm directional impact plot
│   ├── misclassification_analysis.png # Day 8 error spectral curves & safety margins
│   ├── local_prediction_waterfall.png # Day 8 local instance explanations (Mine vs Rock)
│   ├── explainability_report.json # Day 8 structured explainability & error audit report
│   ├── neural_loss_curve.png      # Day 10 MLP Neural Network training loss trajectory
│   ├── ensemble_comparison.png    # Day 10 Multi-paradigm performance benchmark bar chart
│   ├── ensemble_confusion_matrices.png # Day 10 2x2 confusion matrix grid for Day 10 models
│   ├── ensemble_roc_curves.png    # Day 10 Multi-model ROC curves & AUC comparison
│   ├── ensemble_metrics.csv       # Day 10 Comparative metrics spreadsheet
│   └── ensemble_evaluation_report.json # Day 10 Structured benchmark report
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD Pipeline (Test, Lint, Docker Build & GHCR)
├── Dockerfile                     # Lightweight Python production container with healthcheck
├── docker-compose.yml             # Docker Compose orchestration with resource limits
├── .dockerignore                  # Docker build context exclusion rules
├── .gitignore                     # Files and folders to exclude from version control
├── README.md                      # Project documentation and daily tracking
├── requirements.txt               # Project dependencies and libraries (Flask added Day 9)
├── test_environment.py            # Environment and dependency verification script
├── load_data.py                   # Day 2 dataset loading execution script
├── eda.py                         # Day 3 exploratory data analysis execution script
├── preprocess.py                  # Day 4 preprocessing and feature scaling execution script
├── train.py                       # Day 5 model training and cross-validation execution script
├── tune.py                        # Day 6 hyperparameter tuning and diagnostics execution script
├── predict.py                     # Day 7 real-time inference and interactive prediction script
├── explain.py                     # Day 8 model explainability and SHAP analysis execution script
├── app.py                         # Day 9/10 Flask web server, tactical HUD & REST API entry point
├── ensemble.py                    # Day 10 Deep Learning & Stacking/Voting Ensemble runner
└── verify_system.py               # Day 10 Automated End-to-End Pipeline Verification Suite
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

### 10. Real-Time Sonar Signal Inference & Interactive Prediction (Day 7)
Run the real-time interactive prediction CLI to classify new sonar signals:
```bash
python predict.py --test-samples
```
For interactive mode:
```bash
python predict.py --interactive
```

### 11. Model Explainability, SHAP & Error Analysis (Day 8)
Run the comprehensive interpretability and attribution pipeline on the champion model:
```bash
python explain.py
```

#### Key Interpretability Insights:
1. **Permutation Feature Importance (`results/permutation_importance.png`):**
   - Evaluating F1 degradation on unseen test data revealed that frequency bands **`Freq_28`** ($\Delta\text{F1} = +0.0593$), **`Freq_27`** ($\Delta\text{F1} = +0.0588$), **`Freq_22`**, **`Freq_11`**, and **`Freq_12`** are the most sensitive acoustic channels for mine detection.
2. **SHAP Global Feature Attribution (`results/shap_summary.png`, `results/shap_beeswarm.png`):**
   - **`Freq_12`** (mean $|SHAP| = 0.0389$) and **`Freq_11`** (mean $|SHAP| = 0.0300$) exhibit the strongest direct contribution to classifying a signal as an underwater naval mine.
   - Beeswarm distributions show that higher acoustic energy in mid-frequency bands consistently pushes prediction confidence towards the **Mine (Threat)** category.
3. **Local Prediction Waterfalls (`results/local_prediction_waterfall.png`):**
   - Granular feature-by-feature force breakdown explaining individual sonar pings (confident mine vs. confident rock).
4. **Misclassification & Operational Safety Audit (`results/misclassification_analysis.png`, `results/explainability_report.json`):**
   - At the cost-safe operating threshold ($\tau^* = 0.43$):
     - **True Positives (Mines Intercepted):** 22 / 22 (100.0% Recall)
     - **False Negatives (Missed Mines):** **0 (Zero lethal misses)**
     - **False Positives (False Alarms on Rocks):** 3 / 20
   - Spectral deviation curves show that the 3 false alarms occurred on borderline rock signatures with unusually elevated energy in the mid-frequency range.

---

### 12. Interactive Web Application & Tactical Defense Dashboard (Day 9)
Launch the Flask web server to access the full interactive Tactical Sonar Defense Dashboard:
```bash
python app.py
```
Then open your browser at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

#### 🌐 Dashboard Features:
- **Curated Signal Presets:** One-click load of 4 authenticated benchmark signals (Confident Mine, Borderline Mine, Confident Rock, Borderline Rock) from held-out test data.
- **Acoustic Spectral Visualizer:** Real-time canvas chart overlaying the active signal against mean Mine/Rock reference profiles across all 60 frequency bands.
- **Tactical Threat Gauge:** DEFCON-style threat level assessment (DEFCON-1 to DEFCON-4) with probability meters and confidence margin display.
- **Live Decision Threshold Slider ($\tau$):** Sweep the detection threshold from 0.01 to 0.99 in real time. The **Borderline Mine** preset demonstrates the critical difference between $\tau = 0.43$ (intercepted) and $\tau = 0.50$ (missed).
- **On-Demand SHAP Attribution:** Per-signal local feature attribution panel showing the top acoustic frequency bands driving the classification toward Mine or Rock.
- **Batch CSV Upload:** Upload multi-signal CSV files for vectorized batch inference with a structured summary report.

#### 🔌 REST API Endpoints:
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | System status, model version, and loaded artifact diagnostics |
| `GET` | `/api/models` | List all available model architectures (SVM, MLP, Stacking, Voting) |
| `GET` | `/api/samples` | Pre-configured curated test samples and reference spectral profiles |
| `POST` | `/api/predict` | Single 60-band signal classification with optional model & custom threshold |
| `POST` | `/api/explain` | On-demand SHAP local attribution for a given 60-band signal |
| `POST` | `/api/predict_batch` | Multi-signal batch inference from CSV file upload or JSON array |

**Example API usage:**
```bash
# Health check
curl http://127.0.0.1:5000/api/health

# List available model architectures
curl http://127.0.0.1:5000/api/models

# Classify a signal using the Deep Neural Network (MLP)
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [0.028, 0.060, ...], "threshold": 0.43, "model": "mlp_neural_network"}'
```

---

### 13. Deep Learning Neural Networks & Advanced Ensemble Architectures (Day 10)
Execute the Day 10 pipeline to construct, train, benchmark, and serialize the Deep Neural Network (MLP) and Advanced Stacking/Voting Ensembles:
```bash
python ensemble.py
```

#### 🔬 Architectural Implementations:
1. **Deep Multi-Layer Perceptron (MLP) Neural Network (`models/mlp_neural_network.joblib`):**
   - Dense feed-forward architecture designed for 60 continuous acoustic frequency channels:
     $$\text{Input}(60) \longrightarrow \text{Dense}(64, \text{ReLU}) \longrightarrow \text{Dense}(32, \text{ReLU}) \longrightarrow \text{Output}(1, \text{Sigmoid})$$
   - Optimized using Adam optimizer with adaptive learning rates, $L_2$ weight decay ($\alpha = 0.01$), and backpropagation cross-entropy loss tracking.
   - Smooth loss convergence in 80 epochs with final loss $0.0060$ (`results/neural_loss_curve.png`).
2. **Calibrated Soft-Voting Ensemble (`models/voting_ensemble.joblib`):**
   - Blends class posterior probabilities across 4 distinct mathematical algorithmic families:
     - Tuned SVM (RBF margin boundary, weight 2)
     - Tuned Random Forest (non-linear bagged tree partitions, weight 1)
     - Deep Neural Network (distributed continuous representations, weight 1)
     - Tuned Logistic Regression (linear probability baseline, weight 1)
3. **2-Tier Stacking Meta-Classifier (`models/stacking_ensemble.joblib`):**
   - **Tier 1 (Base Learners):** Tuned SVM + Tuned Random Forest + Deep Neural Network (MLP).
   - **Tier 2 (Meta-Learner):** Calibrated Logistic Regression trained on 5-Fold Stratified out-of-fold meta-prediction probabilities.

#### 📊 Multi-Paradigm Performance Benchmark (Unseen Test Set):
| Model Architecture | Paradigm | Test Accuracy | Threat Recall | F1-Score | FN ($\tau = 0.50$) | FN ($\tau^* = 0.43$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tuned SVM (Champion Baseline)** | Margin Kernel | **88.10%** | **90.91%** | **0.8889** | 2 | **0 (Zero Missed)** |
| **Tuned Random Forest** | Bagged Trees | 85.71% | 95.45% | 0.8750 | 1 | 1 |
| **Deep Neural Network (MLP)** | Deep Learning | **90.48%** | **90.91%** | **0.9091** | 2 | 1 |
| **Calibrated Soft-Voting** | Ensemble Voting | **90.48%** | **90.91%** | **0.9091** | 2 | 1 |
| **Stacking Meta-Classifier** | Stacking Meta | **90.48%** | **90.91%** | **0.9091** | 2 | 1 |

#### 🧪 Full System Automated Verification Suite (Day 10)
Run the automated end-to-end verification suite testing all 10 stages:
```bash
python verify_system.py
```
Validates raw data schema, leakage-free scaling, all 13 trained models, dynamic inference switching, zero-false-negative naval safety audit, SHAP explainability, Flask REST endpoints, and all 19 publication visual artifacts.

---

### 14. Docker Containerization & GitHub Actions CI/CD Pipeline (Day 11)

Production containerization and automated continuous integration for the Tactical Sonar Threat Detection platform.

#### 🐳 Docker Containerization
The project includes a hardened, production-ready `Dockerfile` (using `python:3.11-slim`, non-root security principles, and healthcheck probes) and `docker-compose.yml`:

```bash
# Build the Docker image
docker build -t sonar-defense-system:latest .

# Run the container
docker run -p 5000:5000 --name sonar-threat-detector sonar-defense-system:latest

# Or launch via Docker Compose with resource constraints
docker-compose up -d
```

Verify container liveness:
```bash
curl http://localhost:5000/api/health
```

#### 🔄 GitHub Actions CI/CD Pipeline
Continuous integration is managed by [`.github/workflows/ci-cd.yml`](.github/workflows/ci-cd.yml), executing on every push and pull request:
1. **Pipeline & Model Verification (`test`):**
   - Matrix testing across Python `3.10` and `3.11`.
   - Dependency verification (`test_environment.py`).
   - Automated 9-stage end-to-end system verification suite (`verify_system.py`).
2. **Docker Container Deployment (`docker`):**
   - Automated Docker Buildx setup (`docker/setup-buildx-action@v3`) with GitHub Actions cache export/import (`type=gha`).
   - Container liveness and REST API self-test.
   - Pushes production images to GitHub Container Registry (`ghcr.io`) upon merges to `main`.

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
| **Day 7** | Real-Time Sonar Signal Inference & Interactive Prediction System | Completed ✅ |
| **Day 8** | Model Explainability, SHAP Feature Attribution & Error Profile Analysis | Completed ✅ |
| **Day 9** | Interactive Web Application, Tactical Defense Dashboard & Production REST API | Completed ✅ |
| **Day 10** | Deep Learning (Neural Networks - MLP), Advanced Ensemble Learning (Stacking & Soft-Voting) & Full System Verification Suite | Completed ✅ |
| **Day 11** | Containerization & DevOps: Production Dockerfile, Docker Compose Orchestration & GitHub Actions CI/CD Pipeline | Completed ✅ |


---

## 👥 Author & Academic Details
- **Project Title:** Machine Learning-Based Sonar Signal Classification System for Underwater Threat Detection
- **Domain:** Artificial Intelligence & Machine Learning (AI/ML)
