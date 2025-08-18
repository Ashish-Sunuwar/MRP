# MRP
## Sequential Deep Learning & LightGBM for Next-Day Stock Direction
A reproducible research repo for our hybrid forecasting pipeline: a CNN–BiLSTM–GRU encoder that summarizes recent price history, concatenated with same-day tabular features, and classified by LightGBM with probability calibration and a validation-chosen threshold. We evaluate on a large, chronologically split panel (2016–2023) and compare against strong statistical, tree-based, and pure deep baselines.

Research question. Can a hybrid model—deep sequential encoding + gradient-boosted trees—beat traditional time-series models and standalone deep networks at predicting next-day stock direction?

## Overview
Two variants
Price-only hybrid: 60-day price window → 32-D embedding → concat with engineered price features → LightGBM.
Multimodal hybrid: 30-day price window → 32-D embedding → concat with price + FinBERT sentiment aggregates → LightGBM.
Evaluation protocol: Train 2016–2021 → Validate 2022 (early stop, Platt calibration, F1-optimal threshold) → Test 2023.
Headline results (2023):
Hybrid price-only: Acc 0.73, Prec 0.69, Rec 0.85, F1 0.76, AUC 0.82 (best)
Hybrid + sentiment: Acc 0.68, Prec 0.63, Rec 0.91, F1 0.74, AUC 0.80
Key finding: Engineered price features carry most of the signal; sentiment adds a small, regime-dependent lift.

## Repository structure
project-root/
│
├── Experiments/                              # Contains all the codes for the experiments performed along with EDA and SHAP
├── Preprocessing and Feature Engineering/    # Contains all the preprocessing and feature engineering codes for preparing the final dataset for experiments
└── README.md                            # Project documentation

## Setup
## Requirements
Python 3.10+
Recommended: NVIDIA GPU with CUDA 11+ (CPU works but slower)
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
tensorflow>=2.12
lightgbm>=4.0
matplotlib>=3.7
seaborn>=0.12
shap>=0.44

## Data	
Link to the original dataset: https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main
link to the cleaned dataset: https://drive.google.com/drive/folders/1BiMNW7lbCHA0zcwxH3B2k69NiH2dbkRR
“final_dataset_price _only.csv” consists of just the price data
“final_dataset_price_sentiment.csv” consists of price and sentiment data

## Engineered features (formulas)
Backward 1-day return: return_1d[t] = ln(P_t) − ln(P_{t−1})
Forward (label) return: return_1d_fwd[t] = return_1d[t+1] → target[t] = 1(return_1d_fwd[t] > 0)
Lagged returns: return_1d_lag{k}[t] = return_1d[t−k], k ∈ {1,3,5}
7-day trend/risk: return_7d_mean[t] = mean(return_1d[t−7…t−1]); return_7d_std[t] = std(…)
10-day MA: mean of adjusted close over last 10 days
30-day vol: std of return_1d over last 30 days
RSI-14: Wilder’s smoothing of gains/losses over 14 days
Log volume: log(volume)
Sentiment: avg_sentiment = P(pos) − P(neg) ∈ [−1,1]; avg_sentiment_confidence = max{P(pos), P(neg), P(neu)}; rolling 7-day mean/std and counts of positive/negative days (past-only)

Leakage hygiene: all rolling features are past-only (shifted before rolling). Splits are based on the label date (the next trading day).


## Reproducibility
Fixed seeds: Python, NumPy, TensorFlow, LightGBM = 42
Strict chronological splits; label-date used for boundaries to prevent spillover
No cross-set normalization/statistics shared
All rolling features computed from past-only data (shift before rolling)

## Notes & disclaimers
This repository is for research; it is not financial advice.
This research project is strictly for research and educational purposed, and should not be used for commercial purposes.
If you can’t access a GPU, reduce batch size or sequence length to fit memory.

## Contact
-  Ashish Sunuwar- ashish.a.sun@gmail.com
