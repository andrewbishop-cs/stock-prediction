"""
src/evaluation.py

Evaluation utilities for models, including hold-out test evaluation.
"""
import os
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report
import argparse


def evaluate_holdout(features_path: str, model_path: str, test_size: float = 0.2):
    """
    Evaluate a trained model on a hold-out test set.
    Prints AUC and classification report.
    """
    # Load and sort data
    df = pd.read_csv(features_path, parse_dates=['date'])
    df.sort_values(by='date', inplace=True)

    # Prepare X and y
    X = df.drop(['date', 'ticker', 'target', 'text_raw'], axis=1)
    y = df['target']

    # Time-based split
    split_idx = int(len(df) * (1 - test_size))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # Load model
    model = joblib.load(model_path)
    print(f"[EVAL] Loaded model from {model_path}")

     # Predict probabilities
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"[EVAL] Test AUC: {auc:.4f}")

    # Evaluate at multiple thresholds
    from sklearn.metrics import classification_report
    for thresh in [0.50, 0.46]:
        preds = (probs >= thresh).astype(int)
        print(f"\nThreshold = {thresh:.2f}")
        print(classification_report(y_test, preds))

def main():
    parser = argparse.ArgumentParser(description='Model evaluation scripts')
    parser.add_argument('--features',   default='../data/processed/features_final.csv',
                        help='Path to features_final.csv')
    parser.add_argument('--model-path', default='../models/gbm_tuned.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--test-size',  type=float, default=0.2,
                        help='Proportion of data reserved for testing')
    args = parser.parse_args()

    evaluate_holdout(args.features, args.model_path, args.test_size)


if __name__ == '__main__':
    main()
