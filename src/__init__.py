"""
Sonar Signal Classification System
Source Package
"""

from .data_loader import load_sonar_data, display_dataset_info
from .eda import (
    get_dataset_summary,
    check_missing_values,
    check_duplicates,
    analyze_class_distribution,
    compute_statistical_analysis,
    get_features_and_target,
    generate_and_save_visualizations,
    run_full_eda,
)
from .preprocessor import (
    encode_labels,
    split_sonar_data,
    scale_sonar_features,
    save_preprocessed_artifacts,
    load_preprocessed_artifacts,
    SonarPreprocessor,
    run_full_preprocessing,
)
from .model_trainer import (
    get_default_models,
    evaluate_model_cv,
    evaluate_model_test,
    train_and_evaluate_all_models,
    generate_and_save_model_plots,
    save_evaluation_metrics,
    save_trained_models,
    SonarModelTrainer,
    run_full_training,
)
from .tuner import (
    get_hyperparameter_grids,
    tune_all_models,
    optimize_decision_thresholds,
    compute_model_diagnostics,
    generate_and_save_diagnostic_plots,
    save_tuning_artifacts,
    SonarModelTuner,
    run_full_tuning,
)
from .predictor import SonarThreatPredictor

__all__ = [
    "load_sonar_data",
    "display_dataset_info",
    "get_dataset_summary",
    "check_missing_values",
    "check_duplicates",
    "analyze_class_distribution",
    "compute_statistical_analysis",
    "get_features_and_target",
    "generate_and_save_visualizations",
    "run_full_eda",
    "encode_labels",
    "split_sonar_data",
    "scale_sonar_features",
    "save_preprocessed_artifacts",
    "load_preprocessed_artifacts",
    "SonarPreprocessor",
    "run_full_preprocessing",
    "get_default_models",
    "evaluate_model_cv",
    "evaluate_model_test",
    "train_and_evaluate_all_models",
    "generate_and_save_model_plots",
    "save_evaluation_metrics",
    "save_trained_models",
    "SonarModelTrainer",
    "run_full_training",
    "get_hyperparameter_grids",
    "tune_all_models",
    "optimize_decision_thresholds",
    "compute_model_diagnostics",
    "generate_and_save_diagnostic_plots",
    "save_tuning_artifacts",
    "SonarModelTuner",
    "run_full_tuning",
    "SonarThreatPredictor",
]


