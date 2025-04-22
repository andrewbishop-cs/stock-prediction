"""
src/models/threshold_search.py

Search for the optimal probability threshold for classification on the hold-out test set.

python src/models/threshold_search.py \
  --features   data/processed/features_final.csv \
  --model-path models/gbm_tuned.pkl \
  --test-size  0.2 \
  --steps      101


"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, roc_auc_score
import argparse


def threshold_search(features_path, model_path, test_size=0.2, steps=101):
    # Load and sort data
    df = pd.read_csv(features_path, parse_dates=['date'])
    df.sort_values(by='date', inplace=True)

    # Split into train/test
    split_idx = int(len(df) * (1 - test_size))
    X_test = df.drop(['date', 'ticker', 'target', 'text_raw'], axis=1).iloc[split_idx:]
    y_test = df['target'].iloc[split_idx:]

    # Load model
    model = joblib.load(model_path)
    print(f"[THRESH] Loaded model from {model_path}")

    # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]
    print(f"[THRESH] ROC AUC: {roc_auc_score(y_test, probs):.4f}")

    # Create DataFrame of results
    results = pd.DataFrame({
        'date': df['date'].iloc[split_idx:].values,
        'y_true': y_test.values,
        'y_prob': probs
    })
    print(results)

    # Search thresholds
    best_thresh, best_f1 = 0.0, 0.0
    for t in np.linspace(0, 1, steps):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"[THRESH] Best threshold: {best_thresh:.2f} -> F1: {best_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probability threshold search')
    parser.add_argument('--features',  default='../data/processed/features_final.csv', help='Path to features CSV')
    parser.add_argument('--model-path', default='../models/gbm_tuned.pkl', help='Path to trained model')
    parser.add_argument('--test-size', type=float, default=0.2, help='Hold-out fraction')
    parser.add_argument('--steps',     type=int, default=101, help='Number of thresholds to evaluate')
    args = parser.parse_args()

    threshold_search(args.features, args.model_path, args.test_size, args.steps)
