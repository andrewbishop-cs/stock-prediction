"""
src/models/tune_gbm.py

Hyperparameter tuning for the XGBoost model using GridSearchCV
with time-series cross-validation.


python src/models/tune_gbm.py \
  --features   data/processed/features_final.csv \
  --model-out  models/gbm_tuned.pkl \
  --cv-splits  5
"""
import os
import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib


def tune_model(features_path: str, model_out: str, cv_splits: int = 5):
    # Load and sort by date
    df = pd.read_csv(features_path, parse_dates=['date'])
    df.sort_values(by='date', inplace=True)

    # Prepare features and target
    X = df.drop(['date', 'ticker', 'target', 'text_raw'], axis=1)
    y = df['target']

    # TimeSeries cross-validator
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Base model
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )

    # Grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X, y)

    # Results
    print("Best Hyperparameters:", grid.best_params_)
    print(f"Best CV AUC: {grid.best_score_:.4f}")

    # Save best estimator
    best_model = grid.best_estimator_
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(best_model, model_out)
    print(f"Saved tuned model to {model_out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for XGBoost')
    parser.add_argument('--features',   default='../data/processed/features_final.csv',
                        help='Path to features_final.csv')
    parser.add_argument('--model-out',  default='../models/gbm_tuned.pkl',
                        help='Output path for the tuned model')
    parser.add_argument('--cv-splits',  type=int, default=5,
                        help='Number of CV splits for time-series')
    args = parser.parse_args()

    tune_model(args.features, args.model_out, args.cv_splits)
