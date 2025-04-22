"""
src/features.py

Combine embeddings + TA features with target labels for modeling.

python src/features.py \
  --price-dir    data/raw \
  --features-emb data/processed/features_emb.csv \
  --output-csv   data/processed/features_final.csv


"""
import os
import argparse
import pandas as pd
from glob import glob


def compute_targets(price_dir: str) -> pd.DataFrame:
    """
    Compute daily up/down movement per ticker.
    Returns DataFrame with ['date','ticker','target'] where target=1 if today's close > yesterday's close, else 0.
    """
    records = []
    for f in glob(os.path.join(price_dir, '*_prices.csv')):
        ticker = os.path.basename(f).split('_')[0]
        df = pd.read_csv(f, parse_dates=['Date'], index_col='Date')
        df.sort_index(inplace=True)
        # Compute movement: 1 if today close > yesterday close
        df['prev_close'] = df['Close'].shift(1)
        df['target'] = (df['Close'] > df['prev_close']).astype(int)
        # Only keep rows aligned with features (date index)
        target_df = df[['target']].reset_index().rename(columns={'Date': 'date'})
        target_df['ticker'] = ticker
        records.append(target_df[['date','ticker','target']])
    return pd.concat(records, ignore_index=True)


def load_features(features_emb_path: str) -> pd.DataFrame:
    """
    Load features with embeddings (features_emb.csv) and parse date column.
    """
    df = pd.read_csv(features_emb_path, parse_dates=['date'])
    return df


def merge_all(features_emb: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """
    Merge embeddings+TA features with target labels.
    """
    merged = pd.merge(
        features_emb,
        targets,
        on=['date','ticker'],
        how='inner'
    )
    # Drop rows with missing
    merged.dropna(inplace=True)
    return merged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine features and targets for modeling')
    parser.add_argument('--price-dir', default='../data/raw', help='Directory with price CSVs')
    parser.add_argument('--features-emb', required=True, help='Path to features_emb.csv')
    parser.add_argument('--output-csv', default='../data/processed/features_final.csv', help='Output path for final dataset')
    args = parser.parse_args()

    # Compute target labels
    targets = compute_targets(args.price_dir)
    # Load features with embeddings
    feats = load_features(args.features_emb)
    # Merge
    final_df = merge_all(feats, targets)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    final_df.to_csv(args.output_csv, index=False)
    print(f"[FEATURES] Wrote final dataset with {len(final_df)} rows to {args.output_csv}")
