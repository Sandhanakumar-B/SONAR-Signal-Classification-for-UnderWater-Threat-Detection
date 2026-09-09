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
]

