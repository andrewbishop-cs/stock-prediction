# Stock Movement Predictor 🔄

**Financial ML pipeline for predicting next-day stock movements using price data and GDELT news signals.**

---

## 📋 Overview

This project builds a fully automated, end-to-end workflow to:

1. **Ingest** daily stock prices (via yfinance) and structured news data (GDELT mentions & events).
2. **Engineer** technical indicators (RSI, MACD, Bollinger Bands, etc.) and Word2Vec embeddings for text signals.
3. **Train & tune** machine learning models (Logistic Regression, XGBoost) to predict next-day price direction.
4. **Evaluate** performance (AUC, backtest returns, Sharpe ratio).
5. **Deploy** a Streamlit dashboard for live signals and automated retraining.

> 🔧 **Status:** Under active development — ongoing enhancements to data pipelines, model performance, and backtesting rigor.

---

## 🛠️ Tech & Concepts

- **Languages & Libraries:** Python 3.11 · pandas · NumPy · scikit-learn · XGBoost · Gensim (Word2Vec) · Streamlit
- **Data Sources:** yfinance (price), GDELT Global Knowledge Graph (mentions & events)
- **Modeling:** Logistic Regression baseline · Gradient Boosting (XGBoost) · GridSearchCV/Optuna tuning
- **Evaluation Metrics:** ROC AUC, Precision/Recall, Cumulative returns, Sharpe Ratio, Max Drawdown
- **Deployment:** Streamlit dashboard with daily prediction & retraining triggers

---

## 📁 Repository Structure

```
stock-news-predictor/
├── data/
│   ├── raw/           # Price CSVs, mentions/, events/
│   └── processed/     # features.csv, features_emb.csv, features_final.csv
├── models/            # Saved .pkl and embedding .model files
├── notebooks/         # EDA + ROC/PR plotting notebooks
├── src/
│   ├── data_ingest.py # Fetch prices + GDELT mentions & events
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── features.py
│   ├── models/
│   │   ├── logistic.py
│   │   └── gbm.py
│   ├── gbm_tune.py
│   ├── evaluation.py
│   ├── threshold_search.py
│   └── backtest.py
├── app/
│   └── dashboard.py   # Streamlit interface
├── tests/             # Pytest tests for data ingestion
├── requirements.txt
├── README.md          # ← you are here
└── RUN_WORKFLOW.md    # Full step-by-step guide
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
python3.11 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\\Scripts\\activate     # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Ingestion

```bash
python src/data_ingest.py \
  --use-sp500 \
  --start-date 2015-02-19 \
  --end-date   2025-04-22 \
  --output-dir data/raw
```

### 3. Preprocessing

```bash
python src/preprocessing.py \
  --price-dir data/raw \
  --mentions-dir data/raw/mentions \
  --events-dir data/raw/events \
  --output-csv data/processed/features.csv
```

### 4. Train Embeddings

```bash
python src/embeddings.py \
  --features data/processed/features.csv \
  --model-path models/w2v.model \
  --output-csv data/processed/features_emb.csv
```

### 5. Aggregate Features

```bash
python src/features.py \
  --input-csv data/processed/features_emb.csv \
  --output-csv data/processed/features_final.csv
```

### 6. Model Training

```bash
python src/models/logistic.py --features data/processed/features_final.csv --model-out models/logistic.pkl --test-size 0.2
python src/models/gbm.py      --features data/processed/features_final.csv --model-out models/gbm.pkl      --test-size 0.2 --n-estimators 200 --learning-rate 0.05 --max-depth 5
```

### 7. Hyperparameter Tuning

```bash
python src/gbm_tune.py --features data/processed/features_final.csv --model-out models/gbm_tuned.pkl
```

### 8. Evaluation & Threshold Search

```bash
python src/evaluation.py        --features data/processed/features_final.csv --model-path models/gbm_tuned.pkl --test-size 0.2
python src/threshold_search.py  --features data/processed/features_final.csv --model-path models/gbm_tuned.pkl --test-size 0.2
```

### 9. Backtest Strategy

```bash
python src/backtest.py --features data/processed/features_final.csv --model-path models/gbm_tuned.pkl --threshold 0.46
```

### 10. Launch Dashboard

```bash
streamlit run app/dashboard.py -- --features data/processed/features_final.csv --model models/gbm_tuned.pkl
```

