"""
Main execution script for flu forecasting project
Runs data extraction, baseline SARIMA, SIR, and SEIR models
"""

import os
import pandas as pd

from data_extractor import main as extract_main
from baseline import main as baseline_main
from SIR_model import main as sir_main
from SEIR_model import main as seir_main

from utils import calculate_metrics


def run_data_extraction():
    """Run data extraction"""
    print("\n" + "="*80)
    print("STEP 1: DATA EXTRACTION")
    print("="*80)
    extract_main()


def run_baseline_model():
    """Run baseline SARIMA model"""
    print("\n" + "="*80)
    print("STEP 2: BASELINE SARIMA MODEL")
    print("="*80)
    baseline_main()


def run_sir_model():
    """Run SIR epidemiological model"""
    print("\n" + "="*80)
    print("STEP 3: SIR EPIDEMIOLOGICAL MODEL")
    print("="*80)
    sir_main()


def run_seir_model():
    """Run SEIR epidemiological model"""
    print("\n" + "="*80)
    print("STEP 4: SEIR EPIDEMIOLOGICAL MODEL")
    print("="*80)
    seir_main()

def compare_results():
    """Compare results from all models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)

    # Load predictions
    sarima_pred_path = os.path.join('../predictions', 'sarima_predictions.csv')
    sir_pred_path = os.path.join('../predictions', 'sir_predictions.csv')
    seir_pred_path = os.path.join('../predictions', 'seir_predictions.csv')

    sarima_cv_path = os.path.join('../predictions', 'sarima_cv_results.csv')
    sir_cv_path = os.path.join('../predictions', 'sir_cv_results.csv')
    seir_cv_path = os.path.join('../predictions', 'seir_cv_results.csv')

    # Compare CV results
    if os.path.exists(sarima_cv_path) and (os.path.exists(sir_cv_path) or os.path.exists(seir_cv_path)):
        print("\nCross-Validation Results Comparison:")
        print("-" * 80)

        sarima_cv = pd.read_csv(sarima_cv_path)
        sir_cv = pd.read_csv(sir_cv_path) if os.path.exists(sir_cv_path) else None
        seir_cv = pd.read_csv(seir_cv_path) if os.path.exists(seir_cv_path) else None

        for region in ['CA', 'MA', 'NY', 'nat']:
            print(f"\nRegion: {region}")
            print("-" * 40)

            sarima_region = sarima_cv[sarima_cv['region'] == region]
            sir_region = sir_cv[sir_cv['region'] == region] if sir_cv is not None else None
            seir_region = seir_cv[seir_cv['region'] == region] if seir_cv is not None else None

            if not sarima_region.empty:
                print(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'R²':<12}")
                print("-" * 60)

                # SARIMA metrics
                sarima_metrics = {
                    'rmse': sarima_region['rmse'].mean(),
                    'mae': sarima_region['mae'].mean(),
                    'mape': sarima_region['mape'].mean(),
                    'r2': sarima_region['r2'].mean()
                }

                print(f"{'SARIMA':<15} "
                      f"{sarima_metrics['rmse']:<12.4f} "
                      f"{sarima_metrics['mae']:<12.4f} "
                      f"{sarima_metrics['mape']:<12.2f} "
                      f"{sarima_metrics['r2']:<12.4f}")

                # SIR metrics
                if sir_region is not None and not sir_region.empty:
                    sir_metrics = {
                        'rmse': sir_region['rmse'].mean(),
                        'mae': sir_region['mae'].mean(),
                        'mape': sir_region['mape'].mean(),
                        'r2': sir_region['r2'].mean()
                    }

                    print(f"{'SIR':<15} "
                          f"{sir_metrics['rmse']:<12.4f} "
                          f"{sir_metrics['mae']:<12.4f} "
                          f"{sir_metrics['mape']:<12.2f} "
                          f"{sir_metrics['r2']:<12.4f}")

                # SEIR metrics
                if seir_region is not None and not seir_region.empty:
                    seir_metrics = {
                        'rmse': seir_region['rmse'].mean(),
                        'mae': seir_region['mae'].mean(),
                        'mape': seir_region['mape'].mean(),
                        'r2': seir_region['r2'].mean()
                    }

                    print(f"{'SEIR':<15} "
                          f"{seir_metrics['rmse']:<12.4f} "
                          f"{seir_metrics['mae']:<12.4f} "
                          f"{seir_metrics['mape']:<12.2f} "
                          f"{seir_metrics['r2']:<12.4f}")

    # Compare test set results
    if os.path.exists(sarima_pred_path):
        print("\n\nTest Set (2024) Results Comparison:")
        print("-" * 80)

        sarima_pred = pd.read_csv(sarima_pred_path)
        sir_pred = pd.read_csv(sir_pred_path) if os.path.exists(sir_pred_path) else None
        seir_pred = pd.read_csv(seir_pred_path) if os.path.exists(seir_pred_path) else None

        for region in ['CA', 'MA', 'NY', 'nat']:
            print(f"\nRegion: {region}")
            print("-" * 40)

            sarima_region = sarima_pred[sarima_pred['region'] == region]
            sir_region = sir_pred[sir_pred['region'] == region] if sir_pred is not None else None
            seir_region = seir_pred[seir_pred['region'] == region] if seir_pred is not None else None

            if not sarima_region.empty:
                print(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'R²':<12}")
                print("-" * 60)

                # SARIMA metrics
                sarima_metrics = calculate_metrics(
                    sarima_region['true_ili'].values,
                    sarima_region['predicted_ili'].values
                )

                print(f"{'SARIMA':<15} "
                      f"{sarima_metrics['rmse']:<12.4f} "
                      f"{sarima_metrics['mae']:<12.4f} "
                      f"{sarima_metrics['mape']:<12.2f} "
                      f"{sarima_metrics['r2']:<12.4f}")

                # SIR metrics
                if sir_region is not None and not sir_region.empty:
                    sir_metrics = calculate_metrics(
                        sir_region['true_ili'].values,
                        sir_region['predicted_ili'].values
                    )

                    print(f"{'SIR':<15} "
                          f"{sir_metrics['rmse']:<12.4f} "
                          f"{sir_metrics['mae']:<12.4f} "
                          f"{sir_metrics['mape']:<12.2f} "
                          f"{sir_metrics['r2']:<12.4f}")

                # SEIR metrics
                if seir_region is not None and not seir_region.empty:
                    seir_metrics = calculate_metrics(
                        seir_region['true_ili'].values,
                        seir_region['predicted_ili'].values
                    )

                    print(f"{'SEIR':<15} "
                          f"{seir_metrics['rmse']:<12.4f} "
                          f"{seir_metrics['mae']:<12.4f} "
                          f"{seir_metrics['mape']:<12.2f} "
                          f"{seir_metrics['r2']:<12.4f}")

    print("\n" + "="*80)


def main():
    """Main execution function"""
    # Create necessary directories
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../predictions', exist_ok=True)

    print("="*80)
    print("INFLUENZA FORECASTING PIPELINE")
    print("="*80)
    print("\nProject: Disease Spread Modeling with SIR/SEIR")
    print("Regions: CA, MA, NY, National")
    print("Training Period: 2010-2023")
    print("Test Period: 2024")
    print("="*80)

   
    run_data_extraction()
    run_baseline_model()
    run_sir_model()
    run_seir_model()
    compare_results()

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
