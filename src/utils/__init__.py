"""
Utility functions for flu forecasting
"""

from .api_utils import (
    fetch_fluview_data,
    epiweek_to_date,
    parse_epiweek,
    get_population_data,
    get_vaccination_coverage
)

from .preprocessing import (
    preprocess_fluview_data,
    split_train_test,
    create_time_series,
    normalize_ili,
    prepare_cv_folds
)

from .evaluation import (
    calculate_metrics,
    find_peak_week,
    evaluate_peak_prediction,
    cross_validation_summary,
    save_predictions,
    compare_models
)

__all__ = [
    'fetch_fluview_data',
    'epiweek_to_date',
    'parse_epiweek',
    'get_population_data',
    'get_vaccination_coverage',
    'preprocess_fluview_data',
    'split_train_test',
    'create_time_series',
    'normalize_ili',
    'prepare_cv_folds',
    'calculate_metrics',
    'find_peak_week',
    'evaluate_peak_prediction',
    'cross_validation_summary',
    'save_predictions',
    'compare_models'
]
