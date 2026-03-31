#  Quantitative Sports Trading Architecture

**Objective:** Develop a mathematically rigorous Machine Learning pipeline to identify positive Expected Value (EV) in European football betting markets, maximizing long-term Return on Investment (ROI) while strictly preventing data leakage.

##  Project Evolution & Methodology

This repository is structured around three core research phases, demonstrating a progression from baseline statistical models to complex time-series architectures:

### 1. Baseline Exploration (Notebook 01)
* Initial data exploration and statistical modeling using standalone **XGBoost**.
* Implemented `Optuna` for hyperparameter tuning, establishing the foundational feature importance and baseline accuracy metrics.

### 2. Multi-Market Hybrid Ensemble (Notebook 02) - *Best Performing Model*
* **Feature Engineering:** Merged traditional match data with Understat Expected Goals (xG).
* **Mathematical Modeling:** Utilized Poisson distributions to calculate the exact probability matrix for 1X2 and Over/Under 2.5 markets based on dynamic xG inputs.
* **Architecture:** Engineered a hybrid ensemble combining XGBoost decision trees and Keras Neural Networks.
* **Financial Results (Blind Test Set):** * **22.68% ROI** in the Match Odds (1X2) market.
  * **23.08% ROI** in the Goals (Over/Under) market.

### 3. Time-Series & Sequence Modeling (Notebook 03)
* Upgraded the feature space from simple rolling averages to **Exponential Weighted Moving Averages (EWMA)**, capturing the true momentum of teams.
* Transitioned from flat 2D data to 3D tensors to train **Long Short-Term Memory (LSTM)** neural networks and siamese architectures.
* **Financial Results (Blind Test Set):** Achieved a highly stable **16.85% ROI** entirely through sequence recognition.

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                       # Raw CSVs and JSONs (Ignored by Git)
│   └── processed/                 # Cleaned datasets (df_super.csv)
├── models/
│   ├── v1_ensamble_multimercado/  # Best hybrid models (23% ROI)
│   └── v2_ewma_timeseries/        # Time-series models (17% ROI)
├── notebooks/
│   ├── 01_Baseline_XGBoost_Exploration.ipynb
│   ├── 02_Ensemble_xG_Poisson_MultiMarket.ipynb
│   └── 03_TimeSeries_EWMA_and_LSTM.ipynb
├── src/
│   └── oraculo_multimercado_v1.py # Production script with Kelly Criterion
├── README.md                
└── requirements.txt