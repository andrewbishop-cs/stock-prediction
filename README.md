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

