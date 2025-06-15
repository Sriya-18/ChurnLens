# train_model.py

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# 1. Load data
X_train = pd.read_csv("data/features/X_train.csv")
y_train = pd.read_csv("data/features/y_train.csv").values.ravel()
X_test  = pd.read_csv("data/features/X_test.csv")
y_test  = pd.read_csv("data/features/y_test.csv").values.ravel()

# 2. Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# 3. Model training
param_grid = {"C": [0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=1000, class_weight="balanced"),
                    param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train_s, y_train)
model = grid.best_estimator_

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")

# 4. Save evaluation predictions
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]
os.makedirs("data/evaluation", exist_ok=True)
pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_proba": y_proba
}).to_csv("data/evaluation/test_predictions.csv", index=False)

print("✔ Training complete – ROC-AUC:", roc_auc_score(y_test, y_proba).round(3))
