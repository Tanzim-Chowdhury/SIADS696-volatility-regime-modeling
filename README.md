# Short-Term Stock Volatility Regime Modeling

This repository contains the implementation for the SIADS 696 Milestone II project, which focuses on modeling short-term stock volatility regimes using both supervised and unsupervised machine learning methods.

The objective of the project is to predict forward 5-day realized volatility regimes (Low / Mid / High) using historical asset data and market indicators such as the VIX. The project explores both classification-based approaches and unsupervised regime discovery methods.

The analysis includes:
- Feature engineering of realized volatility signals
- Supervised regime classification models
- Unsupervised regime discovery using clustering and Hidden Markov Models
- Evaluation using Macro F1, confusion matrices, and sensitivity analysis


## Main Notebooks

The primary analysis lives in the notebooks directory.

Main notebook:
notebooks/volatility_regime_modeling.ipynb

This notebook contains the full modeling workflow, including:
- Feature engineering
- Regime label construction
- Supervised model training and evaluation
- Unsupervised regime modeling (K-Means and HMM)
- Visualization and analysis of results

Baseline notebook:
notebooks/01_pipeline.ipynb

This notebook served as the initial baseline pipeline and exploratory starting point before the final modeling workflow was developed.


## Project Structure

A simplified overview of the repository structure:

.
├── notebooks/
│   ├── volatility_regime_modeling.ipynb
│   └── 01_pipeline.ipynb
│
├── scripts/
│   ├── fetch_ohlcv.py
│   ├── fetch_vix.py
│   ├── clean_assets.py
│   ├── clean_vix.py
│   ├── merge_datasets.py
│   └── train_baselines.py
│
├── src/
│   ├── pipeline/
│   └── utils/
│
├── data/
│   ├── raw/
│   ├── clean/
│   ├── interim/
│   └── processed/
│
├── figs/
│
└── README.md


### Directory Notes

notebooks/  
Contains exploratory analysis and the final modeling notebook.

scripts/  
Scripts used to fetch, clean, merge, and train models outside of notebooks.

src/  
Reusable Python modules for pipelines, feature engineering, and utilities.

data/  
Organized data storage during pipeline execution.

Data flows through the following stages:
raw → clean → interim → processed

figs/  
Figures generated during analysis and used in the final report.


## Environment Setup

### 1. Create a virtual environment

macOS / Linux
python3 -m venv env
source env/bin/activate

Windows
python -m venv env
env\Scripts\activate


### 2. Install dependencies

Install required packages with pip:

pip install numpy pandas scikit-learn matplotlib seaborn tqdm pyarrow yfinance hmmlearn

If installation issues occur, update pip first:

pip install --upgrade pip setuptools wheel


## Running the Project

Recommended workflow:

Open and run the main notebook:

notebooks/volatility_regime_modeling.ipynb

This notebook executes the full modeling pipeline including:
- Feature construction
- Model training
- Evaluation
- Visualization
- Regime analysis


## Data Pipeline Scripts

Several scripts were used during development to build the dataset and train models. These are included for reference and reproducibility.

Data fetching scripts:
scripts/fetch_ohlcv.py
scripts/fetch_vix.py

These scripts retrieve raw OHLCV market data and VIX data.

Data cleaning scripts:
scripts/clean_assets.py
scripts/clean_vix.py

These scripts standardize and clean the raw datasets.

Dataset construction:
scripts/merge_datasets.py

This merges asset OHLCV data with VIX data into a unified dataset used for modeling.

Training scripts:
scripts/train_baselines.py

This script runs baseline supervised models and generates predictions and evaluation metrics.


## Modeling Overview

Supervised models:
- Logistic Regression
- Random Forest
- HistGradientBoosting

Unsupervised models:
- K-Means clustering
- Gaussian Hidden Markov Model (HMM)

Evaluation metrics include:
- Accuracy
- Macro F1
- Confusion matrices
- Sensitivity analysis


## Data Sources

- Yahoo Finance API
- CBOE Volatility Index (VIX)


## Authors

Tanzim Chowdhury  
Tameem Syed

University of Michigan  
SIADS 696 – Advanced Predictive Modeling