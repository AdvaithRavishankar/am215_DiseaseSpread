"""
Data preprocessing utilities for flu forecasting
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .api_utils import parse_epiweek, epiweek_to_date


def preprocess_fluview_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess FluView data for modeling

    Parameters:
    -----------
    df : pd.DataFrame
        Raw FluView data from API

    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame with additional features
    """
    if df is None or df.empty:
        raise ValueError("Empty or None DataFrame provided")

    # Create a copy to avoid modifying original
    df = df.copy()

    # Parse epiweek into year and week
    df[['year', 'week']] = df['epiweek'].apply(
        lambda x: pd.Series(parse_epiweek(x))
    )

    # Convert epiweek to datetime
    df['date'] = df.apply(
        lambda row: epiweek_to_date(row['year'], row['week']),
        axis=1
    )

    # Sort by region and date
    df = df.sort_values(['region', 'date']).reset_index(drop=True)

    # Handle missing values
    # Use weighted_ili as primary metric (weighted ILI percentage)
    if 'weighted_ili' in df.columns:
        df['ili'] = df['weighted_ili']
    elif 'ili' in df.columns:
        pass  # Use existing ili column
    else:
        raise ValueError("No ILI metric found in data")

    # Fill missing ILI values with forward fill then backward fill
    df['ili'] = df.groupby('region')['ili'].fillna(method='ffill').fillna(method='bfill')

    # Get number of ILI cases (if total_patients available)
    if 'num_ili' in df.columns and 'total_patients' in df.columns:
        df['ili_cases'] = df['num_ili']
    else:
        # Estimate based on ILI percentage (will need population data)
        df['ili_cases'] = df['ili'] * 100  # Placeholder

    # Create season identifier (flu season spans calendar years)
    # Season starts in week 40 and goes to week 20 of next year
    df['season'] = df.apply(lambda row: row['year'] if row['week'] < 40 else row['year'] + 1, axis=1)

    # Add lag features
    for lag in [1, 2, 4]:
        df[f'ili_lag_{lag}'] = df.groupby('region')['ili'].shift(lag)

    # Add rolling average
    df['ili_rolling_4w'] = df.groupby('region')['ili'].rolling(window=4, min_periods=1).mean().reset_index(0, drop=True)

    return df


def split_train_test(
    df: pd.DataFrame,
    train_end_year: int = 2023,
    test_start_year: int = 2024
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed flu data
    train_end_year : int
        Last year to include in training (inclusive)
    test_start_year : int
        First year to use for testing (inclusive, all years >= this will be test)

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Training and testing DataFrames
    """
    train_df = df[df['season'] <= train_end_year].copy()
    test_df = df[df['season'] >= test_start_year].copy()

    return train_df, test_df


def create_time_series(
    df: pd.DataFrame,
    region: str,
    metric: str = 'ili'
) -> pd.Series:
    """
    Create time series for a specific region

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed flu data
    region : str
        Region code
    metric : str
        Metric to extract

    Returns:
    --------
    pd.Series
        Time series with datetime index
    """
    region_df = df[df['region'] == region].copy()
    region_df = region_df.sort_values('date')

    ts = pd.Series(
        region_df[metric].values,
        index=pd.DatetimeIndex(region_df['date']),
        name=f"{region}_{metric}"
    )

    return ts


def normalize_ili(
    df: pd.DataFrame,
    by_population: bool = True,
    population_dict: Optional[dict] = None
) -> pd.DataFrame:
    """
    Normalize ILI cases by population

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed flu data
    by_population : bool
        Whether to normalize by population
    population_dict : dict
        Dictionary mapping region codes to populations

    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized ILI cases
    """
    df = df.copy()

    if by_population and population_dict:
        # Normalize by population (per 100,000)
        df['ili_per_100k'] = df.apply(
            lambda row: (row['ili_cases'] / population_dict.get(row['region'], 1)) * 100000,
            axis=1
        )
    else:
        df['ili_per_100k'] = df['ili_cases']

    return df


def prepare_cv_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    by_season: bool = True
) -> list:
    """
    Prepare cross-validation folds

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed flu data
    n_folds : int
        Number of folds
    by_season : bool
        Whether to split by complete seasons

    Returns:
    --------
    list
        List of (train_indices, test_indices) tuples
    """
    folds = []

    if by_season:
        # Get unique seasons and sort
        seasons = sorted(df['season'].unique())

        # Calculate seasons per fold
        n_seasons = len(seasons)
        seasons_per_fold = max(1, n_seasons // n_folds)

        for i in range(n_folds):
            # Test on one season (or chunk of seasons)
            test_start_idx = i * seasons_per_fold
            test_end_idx = min((i + 1) * seasons_per_fold, n_seasons)

            if test_end_idx > n_seasons:
                break

            test_seasons = seasons[test_start_idx:test_end_idx]
            train_seasons = [s for s in seasons if s not in test_seasons]

            train_idx = df[df['season'].isin(train_seasons)].index
            test_idx = df[df['season'].isin(test_seasons)].index

            if len(train_idx) > 0 and len(test_idx) > 0:
                folds.append((train_idx, test_idx))

    else:
        # Simple time-based splits
        n_samples = len(df)
        fold_size = n_samples // n_folds

        for i in range(n_folds):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_folds - 1 else n_samples

            test_idx = df.index[test_start:test_end]
            train_idx = df.index[~df.index.isin(test_idx)]

            folds.append((train_idx, test_idx))

    return folds
