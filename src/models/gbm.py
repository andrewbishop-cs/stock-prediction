"""
src/models/gbm.py

Train and evaluate a gradient boosting model (XGBoost) and compare against the logistic baseline.

python src/models/gbm.py \
  --features      data/processed/features_final.csv \
  --model-out     models/gbm.pkl \
  --test-size     0.2 \
  --n-estimators  200 \
  --learning-rate 0.05 \
  --max-depth     5


"""
import os
import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib


def train_and_evaluate(features_path: str, model_out: str, test_size: float = 0.2,
                       n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3):
    # Load and sort by date
    df = pd.read_csv(features_path, parse_dates=['date'])
    df.sort_values(by='date', inplace=True)

    # Prepare X/y
    X = df.drop(['date', 'ticker', 'target', 'text_raw'], axis=1)
    y = df['target']

    # Time-based train/test split
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Initialize and train XGBoost
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"[GBM] Test AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    # Save model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"[GBM] Model saved to {model_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train XGBoost gradient boosting model')
    parser.add_argument('--features',    default='../data/processed/features_final.csv',
                        help='Path to features_final.csv')
    parser.add_argument('--model-out',   default='../models/gbm.pkl',
                        help='Path to save trained GBM model')
    parser.add_argument('--test-size',   type=float, default=0.2,
                        help='Fraction of data to reserve for test')
    parser.add_argument('--n-estimators',type=int,   default=100,
                        help='Number of boosting rounds')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate (eta)')
    parser.add_argument('--max-depth',     type=int,   default=3,
                        help='Maximum tree depth')
    args = parser.parse_args()
    train_and_evaluate(
        args.features,
        args.model_out,
        args.test_size,
        args.n_estimators,
        args.learning_rate,
        args.max_depth
    )
