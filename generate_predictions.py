import pandas as pd
import joblib
import os

# Load model & scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load test data
X_test = pd.read_csv("data/features/X_test.csv")
y_test = pd.read_csv("data/features/y_test.csv")

# Create predictions
X_scaled = scaler.transform(X_test)
pred_proba = model.predict_proba(X_scaled)[:, 1]

# Fake metadata for demo
df = X_test.copy()
df["customer_id"] = [f"CUST{i+1}" for i in range(len(df))]
df["pred_churn_proba"] = pred_proba
df["anomaly_flag"] = (df["pred_churn_proba"] > 0.8).astype(int)
df["cohort"] = "2024-Q4"
df["join_date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))

# Save predictions
os.makedirs("data/predictions", exist_ok=True)
df.to_csv("data/predictions/predictions.csv", index=False)

print("✔ Demo predictions.csv created successfully.")

