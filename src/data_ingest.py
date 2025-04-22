"""
src/data_ingest.py

Fetch historical price data and news headlines for specified tickers.

fetch news data:
python src/data_ingest.py \
  --tickers AAPL MSFT GOOGL \
  --start-date 2025-02-13 \
  --end-date   2025-04-21 \
  --news-api-key YOUR_KEY \
  --output-dir data/raw


"""

import os
import argparse
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient


def fetch_price_data(tickers, start_date, end_date, output_dir):
    """
    Download historical OHLCV price data for a list of tickers and save each to CSV.
    """
    # Download grouped by ticker
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Handle single vs. multi-ticker format
    for ticker in tickers:
        try:
            if ticker in data.columns.levels[0]:
                df = data[ticker].copy()
            else:
                df = data.copy()
            file_path = os.path.join(output_dir, f"{ticker}_prices.csv")
            df.to_csv(file_path)
            print(f"[PRICE] Saved {ticker} to {file_path}")
        except Exception as e:
            print(f"[PRICE ERROR] {ticker}: {e}")


def fetch_news_headlines(api_key, tickers, start_date, end_date, output_dir):
    """
    Query NewsAPI for headlines matching each ticker symbol between date range.
    """
    newsapi = NewsApiClient(api_key=api_key)
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tickers:
        try:
            all_articles = []
            # Fetch up to 100 articles per ticker
            response = newsapi.get_everything(
                q=ticker,
                from_param=start_date,
                to=end_date,
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            articles = response.get('articles', [])
            if not articles:
                print(f"[NEWS] No articles for {ticker}")
                continue

            df = pd.DataFrame([{
                'source': a['source']['name'],
                'title': a['title'],
                'description': a['description'],
                'url': a['url'],
                'published_at': a['publishedAt']
            } for a in articles])

            file_path = os.path.join(output_dir, f"{ticker}_news.csv")
            df.to_csv(file_path, index=False)
            print(f"[NEWS] Saved {ticker} news to {file_path}")
        except Exception as e:
            print(f"[NEWS ERROR] {ticker}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch price and news data for tickers.")
    parser.add_argument(
        '--tickers', nargs='+', required=True,
        help='List of ticker symbols, e.g. AAPL MSFT GOOGL'
    )
    parser.add_argument(
        '--start-date', required=True,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end-date', required=True,
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--news-api-key', required=True,
        help='API key for NewsAPI.org'
    )
    parser.add_argument(
        '--output-dir', default='../data/raw',
        help='Directory to save raw CSV files'
    )
    args = parser.parse_args()

    fetch_price_data(
        args.tickers,
        args.start_date,
        args.end_date,
        args.output_dir
    )

    fetch_news_headlines(
        args.news_api_key,
        args.tickers,
        args.start_date,
        args.end_date,
        args.output_dir
    )


if __name__ == '__main__':
    main()
