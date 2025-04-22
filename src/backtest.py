"""
src/backtest.py

Backtest a long-only strategy based on the GBM model's probability signals.
Compute cumulative return, annualized Sharpe ratio, and maximum drawdown.

python src/backtest.py \
  --features   data/processed/features_final.csv \
  --model       models/gbm_tuned.pkl \
  --price-dir  data/raw \
  --threshold  0.46


"""
import pandas as pd
import numpy as np
import joblib
import argparse
import os
from glob import glob


def backtest_strategy(features_path, model_path, price_dir, threshold=0.46):
    # Load features and model
    df = pd.read_csv(features_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    model = joblib.load(model_path)

    # Generate signals
    X = df.drop(['date','ticker','target','text_raw'], axis=1)
    df['prob_up'] = model.predict_proba(X)[:, 1]
    df['signal'] = (df['prob_up'] >= threshold).astype(int)

    # Merge price returns
    returns = []
    for ticker in df['ticker'].unique():
        # load price CSV
        price_file = os.path.join(price_dir, f"{ticker}_prices.csv")
        price = pd.read_csv(price_file, parse_dates=['Date'])
        price.sort_values('Date', inplace=True)
        price['return'] = price['Close'].pct_change()
        # join on date
        temp = df[df['ticker']==ticker][['date','signal']].merge(
            price[['Date','return']],
            left_on='date', right_on='Date', how='left'
        )
        returns.append(temp[['date','signal','return']])
    ret_df = pd.concat(returns, ignore_index=True)

    # Aggregate equal-weight across tickers
    daily = ret_df.groupby('date').agg({
        'return': 'mean',
        'signal': 'mean'
    })
    daily['strategy_ret'] = daily['return'] * daily['signal']

    # Performance metrics
    cum_returns = (1 + daily['strategy_ret']).cumprod() - 1
    cumulative_return = cum_returns.iloc[-1]
    sharpe_ratio = (daily['strategy_ret'].mean() / 
                    daily['strategy_ret'].std()) * np.sqrt(252)
    # max drawdown
    equity = (1 + daily['strategy_ret']).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_drawdown = drawdown.min()

    # Print metrics
    print(f"Cumulative Return: {cumulative_return:.2%}")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest GBM strategy')
    parser.add_argument('--features',  default='data/processed/features_final.csv',
                        help='Path to features_final.csv')
    parser.add_argument('--model',     default='models/gbm_tuned.pkl',
                        help='Path to trained GBM model')
    parser.add_argument('--price-dir', default='data/raw',
                        help='Directory containing <TICKER>_prices.csv')
    parser.add_argument('--threshold', type=float, default=0.46,
                        help='Probability threshold for long signal')
    args = parser.parse_args()

    backtest_strategy(
        args.features, args.model, args.price_dir, args.threshold
    )
