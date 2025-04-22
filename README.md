# stock-prediction

stock-prediction/
├── data/                     # raw & processed data CSVs
├── notebooks/                # EDA + prototyping
├── src/
│   ├── data_ingest.py        # fetch news + price data
│   ├── preprocessing.py      # clean text, compute TA features
│   ├── embeddings.py         # train & save Word2Vec
│   ├── features.py           # aggregate embeddings + indicators
│   ├── models/
│   │   ├── logistic.py       # baseline
│   │   └── gbm.py            # XGBoost/LightGBM pipeline
│   └── evaluation.py         # backtests, AUC calculations
├── app/
│   └── dashboard.py          # Streamlit interface
├── requirements.txt
└── README.md

py -3.11 -m venv venv
source venv/Scripts/activate

fetch news data:
python src/data_ingest.py --tickers AAPL MSFT GOOGL \
  --start-date 2020-01-01 --end-date 2023-12-31 \
  --news-api-key YOUR_API_KEY \
  --output-dir data/raw
  