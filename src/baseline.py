"""
Baseline SARIMA model for influenza forecasting
Implements seasonal ARIMA with 5-fold cross-validation
"""

import os

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


from utils import (
    prepare_cv_folds,
    calculate_metrics,
    find_peak_week,
)

warnings.filterwarnings('ignore')

class SARIMAForecaster:
    """
    SARIMA model for flu forecasting with cross-validation
    """

    def __init__(
        self,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False
    ):
        """
        Initialize SARIMA forecaster

        Parameters:
        -----------
        order : tuple
            (p, d, q) order of ARIMA model
        seasonal_order : tuple
            (P, D, Q, s) seasonal order (s=52 for weekly data)
        enforce_stationarity : bool
            Whether to enforce stationarity
        enforce_invertibility : bool
            Whether to enforce invertibility
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.model_fit = None

    def fit(self, train_data: pd.Series):
        """
        Fit SARIMA model to training data

        Parameters:
        -----------
        train_data : pd.Series
            Training time series
        """
        if len(train_data) == 0:
            raise ValueError("Training data is empty. Cannot fit model.")

        if len(train_data) < 52:  # Need at least one year of weekly data
            print(f"Warning: Training data has only {len(train_data)} observations. Model may not be reliable.")

        self.model = SARIMAX(
            train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        self.model_fit = self.model.fit(disp=False, maxiter=200)
    

    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts

        Parameters:
        -----------
        steps : int
            Number of steps to forecast

        Returns:
        --------
        np.ndarray
            Forecasted values
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before prediction")

        forecast = self.model_fit.forecast(steps=steps)
        return np.array(forecast)

    def get_aic(self) -> float:
        """Get AIC of fitted model"""
        return self.model_fit.aic if self.model_fit else np.inf

    def get_bic(self) -> float:
        """Get BIC of fitted model"""
        return self.model_fit.bic if self.model_fit else np.inf


def run_cross_validation(
    data: pd.DataFrame,
    region: str,
    n_folds: int = 5,
    forecast_horizon: int = 20
) -> pd.DataFrame:
    """
    Run cross-validation for SARIMA model

    Parameters:
    -----------
    data : pd.DataFrame
        Processed flu data
    region : str
        Region to forecast
    n_folds : int
        Number of CV folds
    forecast_horizon : int
        Number of weeks to forecast ahead

    Returns:
    --------
    pd.DataFrame
        Cross-validation results
    """
    print(f"\nRunning {n_folds}-fold cross-validation for {region}...")

    # Filter data for region
    region_data = data[data['region'] == region].copy()
    region_data = region_data.sort_values('date').reset_index(drop=True)

    # Prepare folds
    folds = prepare_cv_folds(region_data, n_folds=n_folds, by_season=True)

    cv_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{len(folds)}...", end=" ")

        train_fold = region_data.loc[train_idx]
        test_fold = region_data.loc[test_idx]

        # Create time series
        train_ts = pd.Series(
            train_fold['ili'].values,
            index=pd.DatetimeIndex(train_fold['date'])
        )

        test_ts = pd.Series(
            test_fold['ili'].values,
            index=pd.DatetimeIndex(test_fold['date'])
        )

        # Fit model
        forecaster = SARIMAForecaster()

        forecaster.fit(train_ts)

        # Predict
        n_test = min(len(test_ts), forecast_horizon)
        predictions = forecaster.predict(steps=n_test)

        # Calculate metrics
        y_true = test_ts.values[:n_test]
        y_pred = predictions[:n_test]

        metrics = calculate_metrics(y_true, y_pred)

        # Find peak weeks
        true_peak_idx, true_peak_val = find_peak_week(y_true)
        pred_peak_idx, pred_peak_val = find_peak_week(y_pred)

        result = {
            'fold': fold_idx + 1,
            'region': region,
            'train_size': len(train_fold),
            'test_size': len(test_fold),
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'mape': metrics['mape'],
            'r2': metrics['r2'],
            'aic': forecaster.get_aic(),
            'bic': forecaster.get_bic(),
            'true_peak_week': true_peak_idx,
            'pred_peak_week': pred_peak_idx,
            'peak_week_error': abs(true_peak_idx - pred_peak_idx)
        }

        cv_results.append(result)
        print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

    return pd.DataFrame(cv_results)


def forecast_test_set(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    region: str,
    forecast_horizon: int = 52
) -> pd.DataFrame:
    """
    Forecast on test set (2024 data)

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data (up to 2023)
    test_data : pd.DataFrame
        Test data (2024)
    region : str
        Region to forecast
    forecast_horizon : int
        Number of weeks to forecast

    Returns:
    --------
    pd.DataFrame
        Predictions with dates
    """
    print(f"\nForecasting {region} for test period (2024)...")

    # Filter by region
    train_region = train_data[train_data['region'] == region].copy()
    test_region = test_data[test_data['region'] == region].copy()

    train_region = train_region.sort_values('date')
    test_region = test_region.sort_values('date')

    # Create time series
    train_ts = pd.Series(
        train_region['ili'].values,
        index=pd.DatetimeIndex(train_region['date'])
    )

    # Fit model
    forecaster = SARIMAForecaster()
    forecaster.fit(train_ts)

    # Forecast
    n_steps = min(len(test_region), forecast_horizon)
    predictions = forecaster.predict(steps=n_steps)

    # Create results DataFrame
    results = pd.DataFrame({
        'date': test_region['date'].values[:n_steps],
        'region': region,
        'true_ili': test_region['ili'].values[:n_steps],
        'predicted_ili': predictions[:n_steps],
        'week': test_region['week'].values[:n_steps],
        'year': test_region['year'].values[:n_steps]
    })

    return results


def main():
    """
    Main execution function for baseline SARIMA model
    """
    print("=" * 60)
    print("Baseline SARIMA Model - Flu Forecasting")
    print("=" * 60)

    # Load processed data
    data_path = os.path.join('../data', 'processed_fluview_data.csv')
    
    print(f"\nLoading data from {data_path}...")
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])

    # Load train/test splits
    train_data = pd.read_csv(os.path.join('../data', 'train_data.csv'))
    test_data = pd.read_csv(os.path.join('../data', 'test_data.csv'))
    train_data['date'] = pd.to_datetime(train_data['date'])
    test_data['date'] = pd.to_datetime(test_data['date'])

    regions = ['CA', 'MA', 'NY', 'nat']

    # Cross-validation
    print("\n" + "=" * 60)
    print("Running 5-Fold Cross-Validation")
    print("=" * 60)

    all_cv_results = []

    for region in regions:
        cv_results = run_cross_validation(
            train_data,
            region=region,
            n_folds=5,
            forecast_horizon=20
        )
        all_cv_results.append(cv_results)

    # Combine CV results
    cv_results_df = pd.concat(all_cv_results, ignore_index=True)

    # Save CV results
    cv_output_path = os.path.join('../predictions', 'sarima_cv_results.csv')
    cv_results_df.to_csv(cv_output_path, index=False)
    print(f"\nCross-validation results saved to: {cv_output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Cross-Validation Summary")
    print("=" * 60)
    summary = cv_results_df.groupby('region')[['rmse', 'mae', 'mape', 'r2']].agg(['mean', 'std'])
    print(summary)

    # Test set forecasting
    print("\n" + "=" * 60)
    print("Test Set Forecasting (2024)")
    print("=" * 60)

    all_predictions = []

    for region in regions:
        predictions = forecast_test_set(
            train_data,
            test_data,
            region=region,
            forecast_horizon=52
        )
        all_predictions.append(predictions)

        # Calculate metrics
        metrics = calculate_metrics(
            predictions['true_ili'].values,
            predictions['predicted_ili'].values
        )
        print(f"\n{region} Test Set Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  R²: {metrics['r2']:.4f}")

    # Save predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    pred_output_path = os.path.join('../predictions', 'sarima_predictions.csv')
    predictions_df.to_csv(pred_output_path, index=False)
    print(f"\nTest predictions saved to: {pred_output_path}")


if __name__ == "__main__":
    # Create predictions directory if it doesn't exist
    os.makedirs('../predictions', exist_ok=True)
    main()
