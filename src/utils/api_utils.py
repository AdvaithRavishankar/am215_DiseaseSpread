"""
Utility functions for fetching data from Delphi Epidata FluView API
"""

import requests
import pandas as pd
from typing import List, Dict, Optional
import time


def fetch_fluview_data(
    regions: List[str],
    epiweeks: str,
    max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Fetch FluView data from Delphi Epidata API

    Parameters:
    -----------
    regions : List[str]
        List of region codes (e.g., ['CA', 'MA', 'NY', 'nat'])
    epiweeks : str
        Epiweek range in format 'YYYYWW-YYYYWW' (e.g., '201001-202452')
    max_retries : int
        Maximum number of retry attempts

    Returns:
    --------
    pd.DataFrame or None
        DataFrame containing flu data or None if request fails
    """
    base_url = "https://api.delphi.cmu.edu/epidata/fluview"

    all_data = []

    for region in regions:
        params = {
            'regions': region,
            'epiweeks': epiweeks
        }

        for attempt in range(max_retries):
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if data['result'] == 1:  # Success
                    all_data.extend(data['epidata'])
                    print(f"Successfully fetched data for {region}")
                    break
                elif data['result'] == -2:  # No data available
                    print(f"No data available for {region}")
                    break
                else:
                    print(f"API returned result code {data['result']} for {region}")

            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for {region}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to fetch data for {region} after {max_retries} attempts")

        time.sleep(0.5)  # Rate limiting

    if all_data:
        return pd.DataFrame(all_data)
    return None


def epiweek_to_date(year: int, week: int) -> pd.Timestamp:
    """
    Convert epiweek (year, week) to approximate date

    Parameters:
    -----------
    year : int
        Year
    week : int
        Week number (1-52/53)

    Returns:
    --------
    pd.Timestamp
        Approximate date for the epiweek
    """
    # Approximate: start of year + (week - 1) * 7 days
    return pd.Timestamp(f'{year}-01-01') + pd.Timedelta(days=(week - 1) * 7)


def parse_epiweek(epiweek: int) -> tuple:
    """
    Parse epiweek integer into year and week

    Parameters:
    -----------
    epiweek : int
        Epiweek in format YYYYWW (e.g., 202001)

    Returns:
    --------
    tuple
        (year, week)
    """
    year = epiweek // 100
    week = epiweek % 100
    return year, week


def get_population_data() -> Dict[str, int]:
    """
    Get population estimates for regions

    Returns:
    --------
    Dict[str, int]
        Dictionary mapping region codes to population estimates
    """
    # Population estimates (approximate, based on 2023 data)
    populations = {
        'CA': 39_029_342,  # California
        'MA': 7_001_399,   # Massachusetts
        'NY': 19_677_151,  # New York
        'nat': 331_000_000  # United States (national)
    }
    return populations


def get_vaccination_coverage() -> Dict[str, float]:
    """
    Get approximate vaccination coverage rates by region

    Returns:
    --------
    Dict[str, float]
        Dictionary mapping region codes to vaccination coverage (0-1)
    """
    # Approximate vaccination coverage rates (based on CDC data)
    # These are approximate averages - ideally would be time-varying
    vaccination_rates = {
        'CA': 0.45,
        'MA': 0.52,
        'NY': 0.47,
        'nat': 0.45
    }
    return vaccination_rates
