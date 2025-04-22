# %% [markdown]
# # ROC and Precision-Recall Curve Analysis
# This notebook will:
# 1. Load the hold-out test set and trained GBM model.
# 2. Compute predicted probabilities.
# 3. Plot the ROC curve.
# 4. Plot the Precision-Recall curve.

# %%
# Imports
import joblib
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# %%
# 1. Load data and model
# Adjust file paths as needed
features_path = '../data/processed/features_final.csv'
model_path = '../models/gbm_tuned.pkl'

# Load features
df = pd.read_csv(features_path, parse_dates=['date'])
df.sort_values('date', inplace=True)

# Create test split
test_size = 0.2
split_idx = int(len(df) * (1 - test_size))
X_test = df.drop(['date','ticker','target','text_raw'], axis=1).iloc[split_idx:]
y_test = df['target'].iloc[split_idx:]

# Load model
model = joblib.load(model_path)

# %%
# 2. Compute predicted probabilities
probs = model.predict_proba(X_test)[:, 1]

# %% [markdown]
# ## ROC Curve
# The ROC curve shows the trade-off between true positive rate and false positive rate.

# %%
# Compute ROC metrics
fpr, tpr, roc_thresholds = roc_curve(y_test, probs)
auc_score = roc_auc_score(y_test, probs)
print(f"ROC AUC: {auc_score:.4f}")

# Plot ROC
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# %% [markdown]
# ## Precision-Recall Curve
# The Precision-Recall curve is useful for imbalanced classes.

# %%
# Compute Precision-Recall metrics
precision, recall, pr_thresholds = precision_recall_curve(y_test, probs)
avg_prec = average_precision_score(y_test, probs)
print(f"Average Precision (PR AUC): {avg_prec:.4f}")
