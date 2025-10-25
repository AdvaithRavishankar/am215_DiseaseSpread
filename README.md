# Influenza Forecasting with Epidemiological Models

## Project Overview

This project implements epidemiological models (SIR/SEIR) to forecast the rise of influenza cases during flu season. The project compares mechanistic compartmental models with baseline statistical forecasting methods (SARIMA) using CDC FluView data.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AdvaithRavishankar/am215_DiseaseSpread.git
cd am215_DiseaseSpread
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
am215_DiseaseSpread/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── api_utils.py          # API data fetching utilities
│   │   ├── preprocessing.py      # Data preprocessing functions
│   │   └── evaluation.py         # Model evaluation metrics
│   ├── data_extractor.py         # Data extraction script
│   ├── baseline.py               # SARIMA baseline model
│   ├── SIR.py                    # SIR/SEIR epidemiological models
│   |── main.py                   # Main execution script
|   |__ eval.ipynb                # Plotting code for analysis 
├── data/                         # Output folder for processed data
├── predictions/                  # Output folder for model predictions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Data Sources

- **CDC FluView**: Historical influenza surveillance data
- **Delphi Epidata FluView API**: Programmatic access to flu data
  - API Documentation: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html


## Installation
### Prerequisites
- Python 3.8 or higher
- pip package manager



## Usage

### Running the Full Pipeline

To run the complete pipeline (data extraction, baseline, SEIR models, and comparison):

```bash
cd src
python main.py
```
## Evaluation Metrics

Models are evaluated using:
- **RMSE** (Root Mean Squared Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average magnitude of errors
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error
- **R²** (Coefficient of Determination): Proportion of variance explained
- **Peak Week Error**: Accuracy in predicting peak flu season timing

## Contributors
Advaith Ravishankar, Darius Sattari, Aviral Misra
