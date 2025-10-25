"""
Data extraction script for CDC FluView data
Fetches data from Delphi Epidata API and saves to CSV
"""

import os
import sys
import pandas as pd

from utils import (
    fetch_fluview_data,
    preprocess_fluview_data,
    normalize_ili,
    get_population_data,
    split_train_test
)


def main():
    """
    Main data extraction workflow
    """
    print("=" * 60)
    print("CDC FluView Data Extraction")
    print("=" * 60)

    # Define regions to fetch
    regions = ['CA', 'MA', 'NY', 'nat']
    print(f"\nFetching data for regions: {', '.join(regions)}")

    # Define epiweek range (2010 to 2025)
    # Format: YYYYWW
    epiweeks = '201001-202552'
    print(f"Epiweek range: {epiweeks}")

    # Fetch data
    print("\nFetching data from Delphi Epidata API...")
    raw_data = fetch_fluview_data(regions=regions, epiweeks=epiweeks)

    if raw_data is None or raw_data.empty:
        print("ERROR: Failed to fetch data or no data available")
        return

    print(f"Successfully fetched {len(raw_data)} records")

    # Save raw data
    raw_data_path = os.path.join('../data', 'raw_fluview_data.csv')
    raw_data.to_csv(raw_data_path, index=False)
    print(f"Raw data saved to: {raw_data_path}")

    # Preprocess data
    print("\nPreprocessing data...")
    
    processed_data = preprocess_fluview_data(raw_data)
    print("Data preprocessing completed")

    # Get population data
    population_dict = get_population_data()

    # Normalize by population
    processed_data = normalize_ili(
        processed_data,
        by_population=True,
        population_dict=population_dict
    )
    print("Data normalized by population")

    # Save processed data
    processed_data_path = os.path.join('../data', 'processed_fluview_data.csv')
    processed_data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to: {processed_data_path}")

    # Split into train and test
    print("\nSplitting data into train/test sets...")
    train_data, test_data = split_train_test(
        processed_data,
        train_end_year=2023,
        test_start_year=2024
    )

    train_path = os.path.join('../data', 'train_data.csv')
    test_path = os.path.join('../data', 'test_data.csv')

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Training data: {len(train_data)} records -> {train_path}")
    print(f"Testing data: {len(test_data)} records -> {test_path}")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    main()
