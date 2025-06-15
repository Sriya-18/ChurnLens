# ChurnLens

ChurnLens helps you find customers who are most at risk of leaving—and gives you an intuitive, browser-based dashboard to explore cohorts, flag anomalies, and take action before it’s too late.

## Why ChurnLens?  
Losing a customer is expensive—and often avoidable. ChurnLens turns your raw customer data into clear, actionable insights so you can prioritize the right accounts, personalize outreach, and keep more people happy.

## What’s Inside?  
- Predictive Model 
  A logistic-regression churn predictor trained on tenure, usage, and spend metrics.  
- Anomaly Alerts 
  Flags customers whose risk spikes beyond their join-month cohort’s norm.  
- Interactive Dashboard
  Built with Streamlit & Altair—filter by cohort or risk threshold, explore histograms of churn probability, scatter plots of high-risk clients, and a cohort heatmap.

## Quick Start

1. Clone the repo  
   bash
    git clone https://github.com/Sriya-18/ChurnLens.git
   cd ChurnLens
   
2.Setup Your Environment
   python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3.Prepare data & train the model
Run these notebooks in order (they’ll produce feature files, train the model, and generate predictions.csv):
notebooks/02-Feature-Engineering.ipynb
notebooks/03-Modelling.ipynb

4. Launch the dashboard
bash
streamlit run notebooks/streamlit_app/app.py
Then open http://localhost:8501 in your browser to explore your churn data.

How It Works
Data Prep: Cleans outliers and engineers key features (tenure, recency, monetary).
Modeling: Fits and tunes a logistic regression to predict individual churn risk.
Visualization: A Streamlit app powered by Altair brings your predictions to life, complete with filters and KPIs.

REPO STRUCTURE 
ChurnLens/
├── data/              
│   ├── raw/           ← your input CSV  
│   ├── features/      ← engineered X_train/X_test, y_train/y_test  
│   └── predictions/   ← model output for dashboard  
├── models/            ← saved model & scaler pickles  
├── notebooks/         ← EDA, feature-engineering, modeling, dashboard code  
└── README.md          


# force redeploy
