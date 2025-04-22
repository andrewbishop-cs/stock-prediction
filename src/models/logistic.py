"""
src/models/logistic.py

Train and evaluate a logistic regression baseline on the feature dataset.


python src/models/logistic.py \
  --features   data/processed/features_final.csv \
  --model-out  models/logistic.pkl \
  --test-size  0.2

"""
import os
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib


def train_and_evaluate(features_path: str, model_out: str, test_size: float = 0.2):
    # Load and sort by date
    df = pd.read_csv(features_path, parse_dates=['date'])
    df.sort_values(by='date', inplace=True)

    # Split into features/target
    X = df.drop(['date', 'ticker', 'target', 'text_raw'], axis=1)
    y = df['target']

    # Time-based train/test split
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"[LOGISTIC] Test AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    # Save model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"[LOGISTIC] Model saved to {model_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train logistic regression baseline')
    parser.add_argument('--features', default='../data/processed/features_final.csv',
                        help='Path to features_final.csv')
    parser.add_argument('--model-out', default='../models/logistic.pkl',
                        help='Path to save trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to reserve for test')
    args = parser.parse_args()
    train_and_evaluate(args.features, args.model_out, args.test_size)
