# Time Series Forecasting: Impact of Target Transformations

This repository contains an ML pipeline for testing the hypothesis: **"Different types of time series (clusters) require different transformations (Log1p, Box-Cox, Diff) to achieve the best forecast quality."**

## Repository structure

```text
├── README.md                 # Project description, structure and launch instructions.
├── requirements.txt          # Dependency versions (for Python 3.13)
├── config.py                 # Global hyperparameters (horizon, window, seasonality, number of clusters, directory paths)
├── Thesis Report.pdf          # Article-like report on the topic (in English)
├── run_experiment.py         # Launching the entire pipeline
│
├── data/                     # Directory for data and artifacts
│   └── m4_daily_ataset.tsf   # Initial dataset
│
├── results/                  # Experimental artifacts
│   ├── metrics.csv           # Summary table with metrics (sMAPE, MASE) for all folds and models
│   └── analysis_results.ipynb  # Jupyter Notebook: EDA, Parameter Justification, Metric Visualization (logs and plots in Russian)
│
└── src/                      # Module source code
    ├── data.py               # Loading and preprocessing the dataset
    ├── eda.py                # Exploratory analysis (stationarity tests, STL, (P)ACF)
    ├── clustering.py         # Statistical feature extraction and clustering
    ├── transforms.py         # Target transformers with inversion (Log1p, Box-Cox, Diff, None)
    ├── baselines.py          # Wrappers for statistical models sktime (ETS, Theta, Naive, ARIMA)
    ├── models.py             # Global CatBoost (MIMO) model with Early Stopping over time
    ├── validation.py         # Walk-Forward Validation Protocol (with Data Leakage Elimination)
    └── metrics.py            # Calculation of metrics (MASE, sMAPE, RMSE)
```

## Key architectural decisions

1. Data Leakage Protection: Strict Walk-Forward cross-validation (`TimeSeriesSplit`) is used. Transformers estimate parameters exclusively on the training set of each fold.
2. Global MIMO Architecture: The ML approach uses `CatBoost` with `MultiRMSE` loss to simultaneously predict the entire horizon vector (MIMO), avoiding error accumulation, overfitting, and inconsistency between prediction steps (as in the Direct strategy).

## Launching the pipeline

### Data preparation
 You need to download the M4 dataset from https://forecastingdata.org/ or Kaggle.   
 Or use the `m4_daily_ataset.tsf` (obtained from https://forecastingdata.org/).

### Running an experiment
You need to run the run_experiments.py script. 

According to the script: 
* Rows will be cleaned and divided into clusters (data, clustering modules).
* Walk-Forward validation will run (baselines, validation, transforms, models, metrics modules).
* Trained CatBoost weights will be saved to the `data/models/` folder.
* Final metrics will be exported to `results/metrics.csv`.

### Analysis and Visualization
To evaluate trained models and the progress of the study, open the analysis_results.ipynb notebook.