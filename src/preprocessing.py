"""
src/preprocessing.py

Clean news text, compute technical indicators on price data, and merge into a single feature DataFrame.

preprocessing:
python src/preprocessing.py \
  --price-dir data/raw \
  --news-dir data/raw \
  --output-csv data/processed/features.csv


if data_ingest raw files cause preprocessing to fail to add any formatted rows, try:

python -> open REPL in root
from src.data_ingest import fetch_price_data
tickers = ['AAPL','MSFT','GOOGL']
fetch_price_data(
    tickers,
    start_date='2025-02-13',
    end_date='2025-04-21',
    output_dir='data/raw'
)

from src.data_ingest import fetch_news_headlines
fetch_news_headlines(
    api_key='9ff18acab7fb4c54b01091ed7b24dd59',
    tickers=tickers,
    start_date='2025-03-22',
    end_date='2025-04-21',
    output_dir='data/raw'
)

then run preprocessing again

"""
import os
import re
import pandas as pd
from glob import glob

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Technical indicators
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('wordnet')

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Lowercase, remove non-alphanum, strip stopwords, and lemmatize.
    """
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+', '', text.lower())
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    cleaned = [LEMMATIZER.lemmatize(tok) for tok in tokens
               if tok not in STOP_WORDS and len(tok) > 2]
    return ' '.join(cleaned)


def load_and_process_news(news_dir: str) -> pd.DataFrame:
    """
    Read all '{ticker}_news.csv' in news_dir, clean text, and aggregate by ticker & date.
    Returns DataFrame with columns: ['date', 'ticker', 'text_raw']
    """
    files = glob(os.path.join(news_dir, '*_news.csv'))
    records = []
    for f in files:
        ticker = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f, parse_dates=['published_at'])
        df['text_raw'] = (
            df['title'].fillna('') + ' ' + df['description'].fillna('')
        ).apply(clean_text)
        df['date'] = pd.to_datetime(df['published_at'].dt.date)
        grouped = df.groupby(['date'])['text_raw'] \
                   .apply(lambda texts: ' '.join(texts)).reset_index()
        grouped['ticker'] = ticker
        records.append(grouped)
    news_df = pd.concat(records, ignore_index=True)
    return news_df


def load_and_compute_ta(price_dir: str) -> pd.DataFrame:
    """
    Read all '{ticker}_prices.csv' in price_dir, compute TA features, shift so features at date t
    predict movement at t+1, and return DataFrame with ['date', 'ticker', features...]
    """
    files = glob(os.path.join(price_dir, '*_prices.csv'))
    ta_records = []
    for f in files:
        ticker = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f, index_col='Date', parse_dates=['Date'])
        df.sort_index(inplace=True)
        # indicators
        df['rsi'] = RSIIndicator(df['Close'], window=14).rsi()
        df['sma_10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
        df['macd'] = MACD(df['Close']).macd()
        bb = BollingerBands(df['Close'], window=20)
        df['bb_hband'] = bb.bollinger_hband()
        df['bb_lband'] = bb.bollinger_lband()
        # shift so today's features predict next-day
        df_feat = df[['rsi','sma_10','macd','bb_hband','bb_lband']].shift(1)
        df_feat = df_feat.reset_index().rename(columns={'Date':'date'})
        df_feat['date'] = pd.to_datetime(df_feat['date'])
        df_feat['ticker'] = ticker
        ta_records.append(df_feat)
    ta_df = pd.concat(ta_records, ignore_index=True)
    return ta_df


def merge_features(news_df: pd.DataFrame, ta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge TA and news DataFrames on ['date','ticker'] using left-join.
    Fills missing news text, drops rows missing TA features.
    """
    news_df['date'] = pd.to_datetime(news_df['date'])
    ta_df['date'] = pd.to_datetime(ta_df['date'])

    merged = pd.merge(
        ta_df,
        news_df,
        on=['date','ticker'],
        how='left'
    )
    # drop rows where TA features are missing (first shifted row)
    merged.dropna(subset=['rsi','sma_10','macd','bb_hband','bb_lband'], inplace=True)
    # fill missing news with empty string
    merged['text_raw'] = merged['text_raw'].fillna('')
    return merged


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument('--price-dir',   default='../data/raw')
    parser.add_argument('--news-dir',    default='../data/raw')
    parser.add_argument('--output-csv',  default='../data/processed/features.csv')
    args = parser.parse_args()

    news = load_and_process_news(args.news_dir)
    ta   = load_and_compute_ta(args.price_dir)
    merged = merge_features(news, ta)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    merged.to_csv(args.output_csv, index=False)
    print(f"[PREPROCESS] Wrote {len(merged)} rows to {args.output_csv}")