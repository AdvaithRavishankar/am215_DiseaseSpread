"""
SEIR Epidemiological Model for Influenza Forecasting
Incorporates vaccination rates and population density
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import odeint
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Dict, Optional


from utils import (
    get_population_data,
    get_vaccination_coverage,
    calculate_metrics,
    find_peak_week,
    prepare_cv_folds,
)

warnings.filterwarnings('ignore')


class SEIRModel:
    """
    SEIR (Susceptible-Exposed-Infected-Recovered) Model with Vaccination
    """

    def __init__(
        self,
        population: int,
        vaccination_rate: float = 0.0,
        initial_exposed: int = 10,
        initial_infected: int = 1
    ):
        """
        Initialize SEIR model

        Parameters:
        -----------
        population : int
            Total population size
        vaccination_rate : float
            Proportion of population vaccinated (0-1)
        initial_exposed : int
            Initial number of exposed individuals
        initial_infected : int
            Initial number of infected individuals
        """
        self.N = population
        self.vaccination_rate = vaccination_rate
        self.initial_exposed = initial_exposed
        self.initial_infected = initial_infected

        # Adjust for vaccination
        self.vaccine_effectiveness = 0.55
        self.protected = int(self.N * vaccination_rate * self.vaccine_effectiveness)

        self.params = None

    def deriv(
        self,
        y: np.ndarray,
        t: float,
        beta: float,
        sigma: float,
        gamma: float
    ) -> list:
        """
        SEIR differential equations

        Parameters:
        -----------
        y : np.ndarray
            Current state [S, E, I, R]
        t : float
            Time
        beta : float
            Transmission rate
        sigma : float
            Incubation rate (1/latent period)
        gamma : float
            Recovery rate

        Returns:
        --------
        list
            Derivatives [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = y
        N = S + E + I + R

        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I

        return [dSdt, dEdt, dIdt, dRdt]

    def simulate(
        self,
        beta: float,
        sigma: float,
        gamma: float,
        days: int = 365
    ) -> np.ndarray:
        """
        Simulate SEIR model

        Parameters:
        -----------
        beta : float
            Transmission rate
        sigma : float
            Incubation rate
        gamma : float
            Recovery rate
        days : int
            Number of days to simulate

        Returns:
        --------
        np.ndarray
            Array of shape (days, 4) containing [S, E, I, R] over time
        """
        # Initial conditions
        E0 = self.initial_exposed
        I0 = self.initial_infected
        R0 = self.protected
        S0 = self.N - E0 - I0 - R0

        y0 = [S0, E0, I0, R0]
        t = np.linspace(0, days, days)

        # Solve ODE
        solution = odeint(self.deriv, y0, t, args=(beta, sigma, gamma))

        return solution

    def fit(
        self,
        observed_cases: np.ndarray,
        method: str = 'differential_evolution'
    ) -> Tuple[float, float, float]:
        """
        Fit SEIR model to observed data

        Parameters:
        -----------
        observed_cases : np.ndarray
            Observed number of infected individuals
        method : str
            Optimization method

        Returns:
        --------
        Tuple[float, float, float]
            Fitted (beta, sigma, gamma) parameters
        """

        def loss(params):
            beta, sigma, gamma = params
            if beta <= 0 or sigma <= 0 or gamma <= 0:
                return 1e10

            try:
                solution = self.simulate(beta, sigma, gamma, days=len(observed_cases))
                predicted = solution[:, 2]  # Infected compartment

                # Scale predicted to match observed
                if np.max(predicted) > 0:
                    scale = np.max(observed_cases) / np.max(predicted)
                    predicted = predicted * scale

                mse = np.mean((observed_cases - predicted) ** 2)
                return mse
            except:
                return 1e10

        # Optimization bounds
        bounds = [(0.01, 2.0), (0.01, 1.0), (0.01, 1.0)]

        if method == 'differential_evolution':
            result = differential_evolution(loss, bounds, seed=42, maxiter=100)
        else:
            x0 = [0.5, 0.2, 0.1]
            result = minimize(loss, x0, method='L-BFGS-B', bounds=bounds)

        self.params = result.x
        return tuple(self.params)

    def forecast(self, days_ahead: int) -> np.ndarray:
        """
        Generate forecast

        Parameters:
        -----------
        days_ahead : int
            Number of days to forecast

        Returns:
        --------
        np.ndarray
            Forecasted infected cases
        """
        if self.params is None:
            raise ValueError("Model must be fitted before forecasting")

        beta, sigma, gamma = self.params
        solution = self.simulate(beta, sigma, gamma, days=days_ahead)

        return solution[:, 2]  # Return infected compartment


def convert_ili_to_cases(ili_percent: float, population: int) -> int:
    """
    Convert ILI percentage to estimated case count

    Parameters:
    -----------
    ili_percent : float
        ILI percentage (0-100)
    population : int
        Total population

    Returns:
    --------
    int
        Estimated number of cases
    """
    return int((ili_percent / 100.0) * population)


def run_seir_cv(
    data: pd.DataFrame,
    region: str,
    n_folds: int = 5,
    forecast_horizon: int = 140  # ~20 weeks in days
) -> pd.DataFrame:
    """
    Run cross-validation for SEIR model

    Parameters:
    -----------
    data : pd.DataFrame
        Processed flu data
    region : str
        Region to forecast
    n_folds : int
        Number of CV folds
    forecast_horizon : int
        Number of days to forecast ahead

    Returns:
    --------
    pd.DataFrame
        Cross-validation results
    """
    print(f"\nRunning {n_folds}-fold cross-validation for {region} (SEIR)...")

    # Get population and vaccination data
    populations = get_population_data()
    vaccination_rates = get_vaccination_coverage()

    population = populations.get(region, 1000000)
    vaccination_rate = vaccination_rates.get(region, 0.45)

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

        # Convert ILI to cases
        train_cases = train_fold['ili'].values * population / 100.0
        test_cases = test_fold['ili'].values * population / 100.0

        # Initialize and fit model
        model = SEIRModel(
            population=population,
            vaccination_rate=vaccination_rate,
            initial_exposed=int(train_cases[0] * 2),
            initial_infected=int(train_cases[0])
        )

        try:
            # Convert to daily data (approximate: repeat weekly values 7 times)
            train_cases_daily = np.repeat(train_cases, 7)

            model.fit(train_cases_daily, method='differential_evolution')

            # Forecast
            n_test_days = min(len(test_cases) * 7, forecast_horizon)
            forecast_daily = model.forecast(days_ahead=n_test_days)

            # Convert back to weekly (average every 7 days)
            forecast_weekly = np.array([
                forecast_daily[i:i+7].mean()
                for i in range(0, len(forecast_daily), 7)
            ])

            # Scale back to ILI percentage
            forecast_ili = (forecast_weekly / population) * 100.0

            # Calculate metrics
            n_test = min(len(test_cases), len(forecast_weekly))
            y_true = test_fold['ili'].values[:n_test]
            y_pred = forecast_ili[:n_test]

            metrics = calculate_metrics(y_true, y_pred)

            # Find peak weeks
            true_peak_idx, _ = find_peak_week(y_true)
            pred_peak_idx, _ = find_peak_week(y_pred)

            result = {
                'fold': fold_idx + 1,
                'region': region,
                'train_size': len(train_fold),
                'test_size': len(test_fold),
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'mape': metrics['mape'],
                'r2': metrics['r2'],
                'true_peak_week': true_peak_idx,
                'pred_peak_week': pred_peak_idx,
                'peak_week_error': abs(true_peak_idx - pred_peak_idx)
            }

            cv_results.append(result)
            print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")

        except Exception as e:
            print(f"Failed: {e}")
            continue

    return pd.DataFrame(cv_results)


def forecast_test_set_seir(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    region: str,
    forecast_horizon: int = 365
) -> pd.DataFrame:
    """
    Forecast on test set using SEIR model

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    region : str
        Region to forecast
    forecast_horizon : int
        Number of days to forecast

    Returns:
    --------
    pd.DataFrame
        Predictions with dates
    """
    print(f"\nForecasting {region} for test period (SEIR)...")

    # Get population and vaccination data
    populations = get_population_data()
    vaccination_rates = get_vaccination_coverage()

    population = populations.get(region, 1000000)
    vaccination_rate = vaccination_rates.get(region, 0.45)

    # Filter by region
    train_region = train_data[train_data['region'] == region].copy()
    test_region = test_data[test_data['region'] == region].copy()

    train_region = train_region.sort_values('date')
    test_region = test_region.sort_values('date')

    # Convert ILI to cases
    train_cases = train_region['ili'].values * population / 100.0

    # Initialize and fit model
    model = SEIRModel(
        population=population,
        vaccination_rate=vaccination_rate,
        initial_exposed=int(train_cases[-1] * 2),
        initial_infected=int(train_cases[-1])
    )

    # Convert to daily data
    train_cases_daily = np.repeat(train_cases, 7)

    model.fit(train_cases_daily, method='differential_evolution')

    # Forecast
    n_test = len(test_region)
    forecast_daily = model.forecast(days_ahead=n_test * 7)

    # Convert back to weekly
    forecast_weekly = np.array([
        forecast_daily[i:i+7].mean()
        for i in range(0, len(forecast_daily), 7)
    ])

    # Scale back to ILI percentage
    forecast_ili = (forecast_weekly / population) * 100.0

    # Create results DataFrame
    n_pred = min(len(test_region), len(forecast_ili))
    results = pd.DataFrame({
        'date': test_region['date'].values[:n_pred],
        'region': region,
        'true_ili': test_region['ili'].values[:n_pred],
        'predicted_ili': forecast_ili[:n_pred],
        'week': test_region['week'].values[:n_pred],
        'year': test_region['year'].values[:n_pred]
    })

    return results


def main():
    """
    Main execution function for SEIR model
    """
    print("=" * 60)
    print("SEIR Epidemiological Model - Flu Forecasting")
    print("=" * 60)

    # Load processed data
    data_path = os.path.join('../data', 'processed_fluview_data.csv')
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please run data_extractor.py first")
        return

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
    print("Running 5-Fold Cross-Validation (SEIR)")
    print("=" * 60)

    all_cv_results = []

    for region in regions:
        cv_results = run_seir_cv(
            train_data,
            region=region,
            n_folds=5,
            forecast_horizon=140
        )
        all_cv_results.append(cv_results)

    # Combine CV results
    if all_cv_results:
        cv_results_df = pd.concat(all_cv_results, ignore_index=True)

        # Save CV results
        cv_output_path = os.path.join('../predictions', 'seir_cv_results.csv')
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
        try:
            predictions = forecast_test_set_seir(
                train_data,
                test_data,
                region=region,
                forecast_horizon=365
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
        except Exception as e:
            print(f"Error forecasting {region}: {e}")

    # Save predictions
    if all_predictions:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        pred_output_path = os.path.join('../predictions', 'seir_predictions.csv')
        predictions_df.to_csv(pred_output_path, index=False)
        print(f"\nTest predictions saved to: {pred_output_path}")


if __name__ == "__main__":
    # Create predictions directory if it doesn't exist
    os.makedirs('../predictions', exist_ok=True)
    main()
