"""
SIR Epidemiological Model for Influenza Forecasting
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


def _get_weekly_case_counts(df: pd.DataFrame, population: int) -> np.ndarray:
    """
    Convert FluView rows into weekly case counts.
    Preference order: reported num_ili -> ili% * num_patients -> ili% of population.
    """
    if 'num_ili' in df.columns and df['num_ili'].notna().any():
        cases = df['num_ili'].fillna(method='ffill').fillna(method='bfill').values
    elif 'num_patients' in df.columns:
        cases = (df['ili'].values / 100.0) * df['num_patients'].values
    else:
        cases = (df['ili'].values / 100.0) * population

    return np.maximum(cases.astype(float), 1e-3)


def _daily_to_weekly_cases(daily_cases: np.ndarray) -> np.ndarray:
    """Aggregate daily cases back to weekly totals by summing each 7-day block."""
    if len(daily_cases) < 7:
        return np.array([])

    n_complete_weeks = len(daily_cases) // 7
    trimmed = daily_cases[: n_complete_weeks * 7]
    return trimmed.reshape(n_complete_weeks, 7).sum(axis=1)


def _cases_to_ili_percent(case_counts: np.ndarray, num_patients: np.ndarray) -> np.ndarray:
    """Convert predicted weekly case counts into ILI percentages using clinic volumes."""
    ili_percent = np.zeros_like(case_counts, dtype=float)
    mask = num_patients > 0
    ili_percent[mask] = (case_counts[mask] / num_patients[mask]) * 100.0
    return ili_percent


def _select_training_window(train_df: pd.DataFrame, test_df: pd.DataFrame, n_seasons: int = 3) -> pd.DataFrame:
    """
    Select the most recent n_seasons training seasons that precede the test period.
    Using multiple seasons provides more robust parameter estimation.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    n_seasons : int
        Number of recent seasons to include (default: 3)
    """
    if train_df.empty or 'season' not in train_df.columns:
        return train_df

    if test_df is None or test_df.empty or 'season' not in test_df.columns:
        available_seasons = sorted(train_df['season'].unique(), reverse=True)
        selected_seasons = available_seasons[:n_seasons]
        return train_df[train_df['season'].isin(selected_seasons)]

    min_test_season = test_df['season'].min()
    candidates = train_df[train_df['season'] < min_test_season]

    if candidates.empty:
        available_seasons = sorted(train_df['season'].unique(), reverse=True)
        selected_seasons = available_seasons[:n_seasons]
        return train_df[train_df['season'].isin(selected_seasons)]

    available_seasons = sorted(candidates['season'].unique(), reverse=True)
    selected_seasons = available_seasons[:n_seasons]
    return candidates[candidates['season'].isin(selected_seasons)]


class SIRModel:
    """
    SIR (Susceptible-Infected-Recovered) Model with Vaccination
    """

    def __init__(
        self,
        population: int,
        vaccination_rate: float = 0.0,
        initial_infected: int = 1,
        use_seasonal_forcing: bool = True
    ):
        """
        Initialize SIR model

        Parameters:
        -----------
        population : int
            Total population size
        vaccination_rate : float
            Proportion of population vaccinated (0-1)
        initial_infected : int
            Initial number of infected individuals
        use_seasonal_forcing : bool
            Whether to use seasonal forcing for transmission rate
        """
        self.N = population
        self.vaccination_rate = vaccination_rate
        self.initial_infected = initial_infected
        self.use_seasonal_forcing = use_seasonal_forcing

        # Adjust susceptible population for vaccination
        # Assume vaccine is ~50-60% effective
        self.vaccine_effectiveness = 0.55
        self.protected = int(self.N * vaccination_rate * self.vaccine_effectiveness)

        self.params = None
        self.scale_factor = 1.0
        self.last_state: Optional[np.ndarray] = None
        self.train_days = 0
        self.train_weeks = 0
        self.seed_cases = None  # average weekly cases used to re-seed a season
        self.seasonal_amplitude = 0.0  # amplitude of seasonal forcing
        self.seasonal_phase = 0.0  # phase shift for seasonal forcing

    def _state_from_seed(self, infected_seed: float) -> np.ndarray:
        """Helper to construct [S, I, R] from a chosen infected count."""
        I0 = max(1.0, float(infected_seed))
        R0 = min(float(self.protected), float(self.N) - I0)
        S0 = max(float(self.N) - I0 - R0, 0.0)
        return np.array([S0, I0, R0], dtype=float)

    def _initial_state(self) -> np.ndarray:
        """Create a safe initial [S, I, R] state using current seed."""
        return self._state_from_seed(self.initial_infected)

    def deriv(self, y: np.ndarray, t: float, beta: float, gamma: float) -> list:
        """
        SIR differential equations with optional seasonal forcing

        Parameters:
        -----------
        y : np.ndarray
            Current state [S, I, R]
        t : float
            Time (in days)
        beta : float
            Base transmission rate
        gamma : float
            Recovery rate

        Returns:
        --------
        list
            Derivatives [dS/dt, dI/dt, dR/dt]
        """
        S, I, R = y
        N = S + I + R

        # Apply seasonal forcing to transmission rate
        # Flu season peaks in winter (around day 0 of year = Jan 1)
        # Use cosine forcing: beta(t) = beta * (1 + amplitude * cos(2*pi*t/365 + phase))
        if self.use_seasonal_forcing:
            seasonal_factor = 1.0 + self.seasonal_amplitude * np.cos(2 * np.pi * t / 365.0 + self.seasonal_phase)
            beta_t = beta * seasonal_factor
        else:
            beta_t = beta

        dSdt = -beta_t * S * I / N
        dIdt = beta_t * S * I / N - gamma * I
        dRdt = gamma * I

        return [dSdt, dIdt, dRdt]

    def simulate(
        self,
        beta: float,
        gamma: float,
        days: int = 365,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate SIR model

        Parameters:
        -----------
        beta : float
            Transmission rate
        gamma : float
            Recovery rate
        days : int
            Number of days to simulate

        Returns:
        --------
        np.ndarray
            Array of shape (days, 3) containing [S, I, R] over time
        """
        # Initial conditions
        if initial_state is None:
            y0 = self._initial_state()
        else:
            y0 = np.asarray(initial_state, dtype=float)

        t = np.linspace(0, days, days)

        # Solve ODE
        solution = odeint(self.deriv, y0, t, args=(beta, gamma))

        return solution

    def _simulate_daily_incidence(
        self,
        beta: float,
        gamma: float,
        days: int,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute daily new infections from the simulated trajectory.
        """
        solution = self.simulate(beta, gamma, days=days, initial_state=initial_state)
        S = solution[:, 0]

        # Daily incidence approximated by decrease in susceptible population
        incidence = np.maximum(-np.diff(S, prepend=S[0]), 0.0)

        return incidence, solution

    def _simulate_weekly_incidence(
        self,
        beta: float,
        gamma: float,
        weeks: int,
        initial_state: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the SIR model and aggregate daily incidence into weekly totals.
        """
        weeks = max(int(weeks), 1)
        days = weeks * 7
        daily_incidence, solution = self._simulate_daily_incidence(
            beta,
            gamma,
            days=days,
            initial_state=initial_state
        )
        weekly_incidence = _daily_to_weekly_cases(daily_incidence)
        return weekly_incidence[:weeks], solution

    @staticmethod
    def _optimal_scale(predicted: np.ndarray, observed: np.ndarray) -> float:
        """Return non-negative least squares scale between predicted and observed series."""
        denom = float(np.dot(predicted, predicted))
        if denom <= 1e-10:
            return 0.0
        scale = float(np.dot(observed, predicted) / denom)
        return max(scale, 0.0)

    def fit(
        self,
        observed_cases: np.ndarray,
        method: str = 'differential_evolution'
    ) -> Tuple[float, float]:
        """
        Fit SIR model to observed data

        Parameters:
        -----------
        observed_cases : np.ndarray
            Observed number of infected individuals (weekly cases)
        method : str
            Optimization method

        Returns:
        --------
        Tuple[float, float]
            Fitted (beta, gamma) parameters
        """

        if len(observed_cases) == 0:
            raise ValueError("Observed cases array is empty")

        self.train_weeks = len(observed_cases)
        self.train_days = max(self.train_weeks * 7, 1)
        # Seed new seasons using the average of the first couple of weeks
        if len(observed_cases) >= 2:
            self.seed_cases = float(np.mean(observed_cases[:2]))
        else:
            self.seed_cases = float(observed_cases[0])
        self.initial_infected = max(1.0, self.seed_cases / 7.0)

        def loss(params):
            if self.use_seasonal_forcing:
                beta, gamma, seasonal_amp, seasonal_phase = params
                if beta <= 0 or gamma <= 0 or seasonal_amp < 0 or seasonal_amp > 0.5:
                    return 1e10
                self.seasonal_amplitude = seasonal_amp
                self.seasonal_phase = seasonal_phase
            else:
                beta, gamma = params
                if beta <= 0 or gamma <= 0:
                    return 1e10

            try:
                predicted, _ = self._simulate_weekly_incidence(
                    beta,
                    gamma,
                    weeks=self.train_weeks
                )

                scale = self._optimal_scale(predicted, observed_cases)
                adjusted = predicted * scale

                mse = np.mean((observed_cases - adjusted) ** 2)

                # Add regularization to prevent overfitting
                regularization = 0.001 * (beta**2 + gamma**2)
                return mse + regularization
            except:
                return 1e10

        # Optimization bounds
        if self.use_seasonal_forcing:
            # beta, gamma, seasonal_amplitude, seasonal_phase
            bounds = [(0.01, 2.0), (0.01, 1.0), (0.0, 0.5), (-np.pi, np.pi)]
            x0 = [0.5, 0.1, 0.2, 0.0]
        else:
            bounds = [(0.01, 2.0), (0.01, 1.0)]
            x0 = [0.5, 0.1]

        if method == 'differential_evolution':
            result = differential_evolution(loss, bounds, seed=42, maxiter=150, workers=1)
        else:
            result = minimize(loss, x0, method='L-BFGS-B', bounds=bounds)

        if self.use_seasonal_forcing:
            beta, gamma, seasonal_amp, seasonal_phase = result.x
            self.seasonal_amplitude = seasonal_amp
            self.seasonal_phase = seasonal_phase
            self.params = (beta, gamma)
        else:
            self.params = result.x

        beta, gamma = self.params

        predicted, fitted_solution = self._simulate_weekly_incidence(
            beta,
            gamma,
            weeks=self.train_weeks
        )
        self.scale_factor = max(self._optimal_scale(predicted, observed_cases), 1e-6)
        self.last_state = fitted_solution[-1]

        return tuple(self.params)

    def forecast(
        self,
        days_ahead: int,
        continue_from_last: bool = True,
        restart_seed_cases: Optional[float] = None,
        use_moderate_restart: bool = True
    ) -> np.ndarray:
        """
        Generate forecast

        Parameters:
        -----------
        days_ahead : int
            Number of days to forecast
        continue_from_last : bool
            If True, continue from last fitted state (better for in-season forecasts)
        restart_seed_cases : Optional[float]
            Override seed cases for new season
        use_moderate_restart : bool
            If True, use a moderate restart that maintains some infected population

        Returns:
        --------
        np.ndarray
            Forecasted infected cases (daily)
        """
        if self.params is None:
            raise ValueError("Model must be fitted before forecasting")

        beta, gamma = self.params

        if continue_from_last and self.last_state is not None:
            # Continue from the last state (in-season forecasting)
            initial_state = self.last_state
        elif use_moderate_restart:
            # Use a moderate restart: keep some proportion of infected from training end
            if self.last_state is not None:
                S_end, I_end, R_end = self.last_state
                # Use 30% of ending infected as starting infected, reset susceptibles
                I0 = max(1.0, I_end * 0.3)
            else:
                I0 = max(1.0, self.seed_cases / 7.0 if self.seed_cases else self.initial_infected)

            # Refresh susceptible pool for new season, maintain recovered
            R0 = min(float(self.protected), float(self.N) - I0)
            S0 = max(float(self.N) - I0 - R0, float(self.N) * 0.5)  # at least 50% susceptible
            initial_state = np.array([S0, I0, R0], dtype=float)
        else:
            # Full restart from seed
            if restart_seed_cases is not None:
                infected_seed = max(1.0, restart_seed_cases / 7.0)
            elif self.seed_cases is not None:
                infected_seed = max(1.0, self.seed_cases / 7.0)
            else:
                infected_seed = self.initial_infected
            initial_state = self._state_from_seed(infected_seed)

        incidence, solution = self._simulate_daily_incidence(
            beta,
            gamma,
            days=days_ahead,
            initial_state=initial_state
        )
        self.last_state = solution[-1]

        return incidence * self.scale_factor  # Return daily new infections


def run_sir_cv(
    data: pd.DataFrame,
    region: str,
    n_folds: int = 5,
    forecast_horizon: int = 140  # ~20 weeks in days
) -> pd.DataFrame:
    """
    Run cross-validation for SIR model

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
    print(f"\nRunning {n_folds}-fold cross-validation for {region} (SIR)...")

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

        # Focus fit on the last available season prior to the test period
        train_season_df = _select_training_window(train_fold, test_fold)

        # Convert ILI metrics to estimated weekly cases
        train_cases = _get_weekly_case_counts(train_season_df, population)
        test_cases = _get_weekly_case_counts(test_fold, population)

        # Initialize and fit model
        model = SIRModel(
            population=population,
            vaccination_rate=vaccination_rate,
            initial_infected=int(max(1.0, np.ceil(train_cases[0] / 7.0)))
        )

        try:
            model.fit(train_cases, method='differential_evolution')

            # Forecast using moderate restart for better continuity
            total_days = max(len(test_cases) * 7, 1)
            forecast_daily = np.maximum(
                model.forecast(
                    days_ahead=total_days,
                    continue_from_last=False,
                    use_moderate_restart=True
                ),
                0.0
            )

            # Convert back to weekly case counts
            forecast_weekly_cases = _daily_to_weekly_cases(forecast_daily)

            # Convert to ILI percentage space for evaluation
            n_test = min(len(test_cases), len(forecast_weekly_cases))
            if n_test == 0:
                raise ValueError("No weekly forecasts produced for evaluation")

            if 'num_patients' in test_fold.columns:
                test_patients = test_fold['num_patients'].values[:n_test]
            else:
                test_patients = np.full(n_test, population)

            y_pred_cases = forecast_weekly_cases[:n_test]
            y_pred = _cases_to_ili_percent(y_pred_cases, test_patients)
            y_true = test_fold['ili'].values[:n_test]

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


def forecast_test_set_sir(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    region: str,
    forecast_horizon: int = 1  # 1 week for rolling forecast (more sensitive to changes)
) -> pd.DataFrame:
    """
    Forecast test set using SIR model with rolling 1-week predictions

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    region : str
        Region to forecast
    forecast_horizon : int
        Number of weeks to forecast ahead (default: 1)

    Returns:
    --------
    pd.DataFrame
        Predictions with dates
    """
    print(f"\nForecasting {region} for test period (SIR) with rolling {forecast_horizon}-week predictions...")

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

    # Restrict training to the final season before the test period
    train_region = _select_training_window(train_region, test_region)

    # Convert ILI metrics to estimated weekly cases
    train_cases = _get_weekly_case_counts(train_region, population)
    test_cases = _get_weekly_case_counts(test_region, population)

    # Initialize and fit model
    model = SIRModel(
        population=population,
        vaccination_rate=vaccination_rate,
        initial_infected=int(max(1.0, np.ceil(train_cases[0] / 7.0)))
    )

    model.fit(train_cases, method='differential_evolution')

    # Rolling window forecasting: predict 1 week, then update with actual data
    n_test = len(test_region)
    all_predictions = []

    for i in range(n_test):
        # Predict 1 week ahead
        forecast_daily = np.maximum(
            model.forecast(
                days_ahead=7,
                continue_from_last=True,
                use_moderate_restart=False
            ),
            0.0
        )

        # Sum the 7 days to get weekly prediction
        weekly_prediction = np.sum(forecast_daily)
        all_predictions.append(weekly_prediction)

        # Update the model state using the actual observed data for this week
        # This keeps the model anchored to reality
        if i < len(test_cases):
            actual_cases = test_cases[i]
            # Simulate one week with the actual case count to update internal state
            # We do this by temporarily adjusting and running the model
            beta, gamma = model.params

            # Get current state and update infected based on actual data
            if model.last_state is not None:
                S, I, R = model.last_state
                # Update I based on actual weekly cases (roughly)
                I_actual = max(1.0, actual_cases / 7.0)
                # Adjust S and R to maintain population balance
                R = min(R + (I - I_actual), population - I_actual)
                S = max(population - I_actual - R, population * 0.1)
                model.last_state = np.array([S, I_actual, R], dtype=float)

    # Convert predictions to ILI percentages
    if 'num_patients' in test_region.columns:
        test_patients = test_region['num_patients'].values[:n_test]
    else:
        test_patients = np.full(n_test, population)

    forecast_cases = np.array(all_predictions[:n_test])
    forecast_ili = _cases_to_ili_percent(forecast_cases, test_patients)

    # Create results DataFrame
    results = pd.DataFrame({
        'date': test_region['date'].values[:n_test],
        'region': region,
        'true_ili': test_region['ili'].values[:n_test],
        'predicted_ili': forecast_ili[:n_test],
        'predicted_cases': forecast_cases,
        'week': test_region['week'].values[:n_test],
        'year': test_region['year'].values[:n_test]
    })

    return results


def main():
    """
    Main execution function for SIR model
    """
    print("=" * 60)
    print("SIR Epidemiological Model - Flu Forecasting")
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

    regions = ['nat']

    # Cross-validation
    print("\n" + "=" * 60)
    print("Running 5-Fold Cross-Validation (SIR)")
    print("=" * 60)

    all_cv_results = []

    for region in regions:
        cv_results = run_sir_cv(
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
        cv_output_path = os.path.join('../predictions', 'sir_cv_results.csv')
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
    print("Test Set Forecasting (2024-2025)")
    print("=" * 60)

    all_predictions = []

    for region in regions:
        try:
            predictions = forecast_test_set_sir(
                train_data,
                test_data,
                region=region
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
        pred_output_path = os.path.join('../predictions', 'sir_predictions.csv')
        predictions_df.to_csv(pred_output_path, index=False)
        print(f"\nTest predictions saved to: {pred_output_path}")


if __name__ == "__main__":
    # Create predictions directory if it doesn't exist
    os.makedirs('../predictions', exist_ok=True)
    main()
