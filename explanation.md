# Disease Spread Models - Data and Implementation Explanation

## Overview

This project implements three different models for forecasting influenza-like illness (ILI) spread:
1. **SARIMA (Seasonal ARIMA)** - Statistical baseline model
2. **SIR Model** - Compartmental epidemiological model
3. **SEIR Model** - Extended compartmental model with exposed state

All models use data from the **Delphi Epidata FluView API** (CDC's influenza surveillance data).

---

## Data Source

### Primary Data Source: Delphi Epidata FluView API

**API Endpoint:** `https://api.delphi.cmu.edu/epidata/fluview`

**Data Retrieved:**
- ILI percentage (weighted percentage of influenza-like illness visits)
- Epiweeks (epidemiological weeks in format YYYYWW)
- Regions: California (CA), Massachusetts (MA), New York (NY), and National (nat)
- Time period: 2010-2025

**Implementation:** See [api_utils.py](src/utils/api_utils.py)

### Supplementary Data

**Population Estimates** (2023 data):
- California: 39,029,342
- Massachusetts: 7,001,399
- New York: 19,677,151
- National: 331,000,000

**Vaccination Coverage Rates** (CDC-based estimates):
- California: 45%
- Massachusetts: 52%
- New York: 47%
- National: 45%

**Source:** [api_utils.py:113-149](src/utils/api_utils.py#L113-L149)

---

## 1. SARIMA Model (Baseline)

### File Location
[src/baseline.py](src/baseline.py)

### What It Predicts
The SARIMA model predicts **future ILI percentage values** (weekly influenza-like illness percentage).

### Model Inputs

**Training Data:**
- **Time series of ILI percentages** from 2010-2023
- Data indexed by date (weekly observations)
- Minimum requirement: 52 weeks (1 year) of data
- Test period: 2024-2025

**Model Parameters:**
- `order=(1, 1, 1)`: ARIMA parameters (p, d, q)
  - p=1: Autoregressive order
  - d=1: Differencing order
  - q=1: Moving average order
- `seasonal_order=(1, 1, 1, 52)`: Seasonal parameters (P, D, Q, s)
  - P=1: Seasonal autoregressive order
  - D=1: Seasonal differencing order
  - Q=1: Seasonal moving average order
  - s=52: Seasonal period (52 weeks per year)

### How It Works

**Model Type:** Seasonal AutoRegressive Integrated Moving Average (SARIMA)

**What the model learns:**
- Historical patterns in ILI percentages
- Seasonal trends (52-week yearly cycles)
- Temporal dependencies between weeks

**Prediction Method:**
- Fitted using maximum likelihood estimation
- Generates forecasts for next N weeks based on historical patterns
- Uses `statsmodels.tsa.statespace.sarimax.SARIMAX` implementation

**Validation:**
- 5-fold cross-validation with seasonal splitting
- Training period: 2010-2023
- Test period: 2024-2025
- Forecast horizon: 20 weeks (cross-validation), 52+ weeks (test set)
---

## 2. SIR Model

### File Location
[src/SIR_model.py](src/SIR_model.py)

### What It Predicts
The SIR model predicts **infected case counts** over time using a compartmental disease model.

### Model Inputs

**Population-based Parameters:**
- `population`: Total population size (region-specific)
- `vaccination_rate`: Proportion vaccinated (0-1, default 0.0)
- `vaccine_effectiveness`: Assumed 55% effectiveness (~0.55)
- `initial_infected`: Initial infected individuals (default 1)

**Data Used for Fitting:**
- **ILI percentages** converted to case counts: `cases = (ili_percent / 100.0) * population`
- Weekly data converted to daily (each week repeated 7 times)
- Training period: 2010-2023
- Test period: 2024-2025

**Data Sources:**
- ILI percentages: Delphi FluView API
- Population: `get_population_data()` in [api_utils.py:113-129](src/utils/api_utils.py#L113-L129)
- Vaccination coverage: `get_vaccination_coverage()` in [api_utils.py:132-149](src/utils/api_utils.py#L132-L149)

### How It Works

**Model Type:** Compartmental ODE-based epidemiological model

**Compartments:**
- **S (Susceptible)**: People who can get infected
- **I (Infected)**: People currently infected
- **R (Recovered)**: People who have recovered or been vaccinated

**Differential Equations:**
```
dS/dt = -beta * S * I / N
dI/dt = beta * S * I / N - gamma * I
dR/dt = gamma * I
```

**Parameters:**
- **beta**: Transmission rate (fitted from data)
- **gamma**: Recovery rate (fitted from data)
- **N**: Total population

**Initial Conditions:**
- S(0) = N × (1 - vaccination_rate × vaccine_effectiveness) - initial_infected
- I(0) = initial_infected
- R(0) = N × vaccination_rate × vaccine_effectiveness

**Fitting Process:**
1. Converts ILI percentages to case counts
2. Uses optimization (differential evolution or L-BFGS-B) to find best beta and gamma
3. Minimizes mean squared error between simulated and observed infected cases
4. Optimization bounds: beta ∈ [0.01, 2.0], gamma ∈ [0.01, 1.0]

---

## 3. SEIR Model

### File Location
[src/SEIR_model.py](src/SEIR_model.py)

### What It Predicts
The SEIR model predicts **infected case counts** over time, accounting for an incubation period.

### Model Inputs

**Population-based Parameters:**
- `population`: Total population size (region-specific)
- `vaccination_rate`: Proportion vaccinated (0-1, default 0.0)
- `vaccine_effectiveness`: Assumed 55% effectiveness (~0.55)
- `initial_exposed`: Initial exposed individuals (default 10)
- `initial_infected`: Initial infected individuals (default 1)

**Data Used for Fitting:**
- **ILI percentages** converted to case counts: `cases = (ili_percent / 100.0) * population`
- Weekly data converted to daily (each week repeated 7 times)
- Training period: 2010-2023
- Test period: 2024-2025

**Data Sources:**
- ILI percentages: Delphi FluView API
- Population: `get_population_data()` in [api_utils.py:113-129](src/utils/api_utils.py#L113-L129)
- Vaccination coverage: `get_vaccination_coverage()` in [api_utils.py:132-149](src/utils/api_utils.py#L132-L149)

### How It Works

**Model Type:** Extended compartmental ODE-based epidemiological model

**Compartments:**
- **S (Susceptible)**: People who can get infected
- **E (Exposed)**: People infected but not yet infectious (incubation period)
- **I (Infected)**: People currently infectious
- **R (Recovered)**: People who have recovered or been vaccinated

**Differential Equations:**
```
dS/dt = -beta * S * I / N
dE/dt = beta * S * I / N - sigma * E
dI/dt = sigma * E - gamma * I
dR/dt = gamma * I
```

**Parameters:**
- **beta**: Transmission rate (fitted from data)
- **sigma**: Incubation rate = 1 / latent_period (fitted from data)
- **gamma**: Recovery rate (fitted from data)
- **N**: Total population

**Initial Conditions:**
- S(0) = N × (1 - vaccination_rate × vaccine_effectiveness) - initial_exposed - initial_infected
- E(0) = initial_exposed
- I(0) = initial_infected
- R(0) = N × vaccination_rate × vaccine_effectiveness

**Fitting Process:**
1. Converts ILI percentages to case counts
2. Uses optimization (differential evolution or L-BFGS-B) to find best beta, sigma, and gamma
3. Minimizes mean squared error between simulated and observed infected cases
4. Optimization bounds:
   - beta ∈ [0.01, 2.0]
   - sigma ∈ [0.01, 1.0]
   - gamma ∈ [0.01, 1.0]

## Evaluation Metrics
All models are evaluated using:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **Peak Week Error** (difference between predicted and actual peak week)
