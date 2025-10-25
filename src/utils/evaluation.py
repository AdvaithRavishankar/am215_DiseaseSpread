"""
Model evaluation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics

    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns:
    --------
    Dict[str, float]
        Dictionary of metric names and values
    """
    # Handle cases where predictions might have different lengths
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan
        }

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
        'r2': r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    }

    return metrics


def find_peak_week(time_series: np.ndarray, dates: pd.DatetimeIndex = None) -> Tuple[int, float]:
    """
    Find the peak week in a time series

    Parameters:
    -----------
    time_series : np.ndarray
        Time series data
    dates : pd.DatetimeIndex
        Corresponding dates

    Returns:
    --------
    Tuple[int, float]
        (peak_week_index, peak_value)
    """
    peak_idx = np.argmax(time_series)
    peak_value = time_series[peak_idx]

    return peak_idx, peak_value


def evaluate_peak_prediction(
    true_peak_week: int,
    pred_peak_week: int,
    tolerance: int = 1
) -> Dict[str, any]:
    """
    Evaluate peak week prediction accuracy

    Parameters:
    -----------
    true_peak_week : int
        True peak week index
    pred_peak_week : int
        Predicted peak week index
    tolerance : int
        Acceptable week difference

    Returns:
    --------
    Dict[str, any]
        Dictionary with peak evaluation metrics
    """
    week_difference = abs(true_peak_week - pred_peak_week)

    return {
        'peak_week_error': week_difference,
        'within_tolerance': week_difference <= tolerance
    }


def cross_validation_summary(cv_results: List[Dict]) -> pd.DataFrame:
    """
    Summarize cross-validation results

    Parameters:
    -----------
    cv_results : List[Dict]
        List of dictionaries containing metrics for each fold

    Returns:
    --------
    pd.DataFrame
        Summary statistics across folds
    """
    df = pd.DataFrame(cv_results)

    summary = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max()
    })

    return summary


def save_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    output_path: str
) -> None:
    """
    Save predictions to CSV file

    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame containing predictions
    model_name : str
        Name of the model
    output_path : str
        Path to save predictions
    """
    filename = f"{output_path}/{model_name}_predictions.csv"
    predictions.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")


def compare_models(results_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare multiple models' results

    Parameters:
    -----------
    results_dict : Dict[str, pd.DataFrame]
        Dictionary mapping model names to results DataFrames

    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    comparison = []

    for model_name, results in results_dict.items():
        if 'rmse' in results.columns:
            comparison.append({
                'model': model_name,
                'mean_rmse': results['rmse'].mean(),
                'mean_mae': results['mae'].mean(),
                'mean_mape': results['mape'].mean(),
                'mean_r2': results['r2'].mean()
            })

    return pd.DataFrame(comparison)
