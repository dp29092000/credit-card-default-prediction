# 💳 Credit Card Default Prediction

## Problem Statement
Predict whether a credit card customer will default on their next payment using machine learning. Built on the UCI Credit Card Default Dataset (30,000 customers, Taiwan, 2005).

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Imbalanced-learn, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Environment:** Google Colab, VS Code

## Project Structure
- `app.py` — Home page
- `pages/1_EDA.py` — Exploratory Data Analysis
- `pages/2_Model_Performance.py` — Model comparison and metrics
- `pages/3_Predict.py` — Live prediction interface
- `xgb_fine_tuned.pkl` — Saved final model
- `median_values.csv` — Feature medians for default inputs
- `UCI_Credit_Card.csv` — Dataset

## Key Results

| Model | Recall | FP | TP |
|-------|--------|----|----|
| Logistic Regression | 62% | 1399 | 828 |
| RF SMOTE (0.29) | 73% | 1752 | 966 |
| XGBoost Tuned (0.54) | 59% | 782 | 784 |
| LGBM SMOTE (0.38) | 60% | 1046 | 790 |

**Final Model: XGBoost Tuned (Threshold 0.54)** — best balance of precision and recall.

## Feature Engineering
- `Pay_Trend` — sum of PAY_1 to PAY_6, captures payment behavior over 6 months
- `average_credit_utilization` — mean bill amount / credit limit
- `Pay_Ratio` — total payment / total bill amount
- `Has_Negative_Bill` — binary flag for credit balance months

## Key Insights
- **PAY_1 is the dominant predictor at 33% importance** — most recent payment status strongly predicts default
- **Customers with PAY_1 > 1 (2+ month delay) have >50% default rate**
- **Pay_Trend (engineered feature) ranked 2nd at 15%** — validating feature engineering effort
- **Lower credit limit customers default more** — confirmed by boxplot analysis
- **Age is not a significant predictor** — similar distribution across defaulters and non-defaulters
- **Class imbalance handled via SMOTE and scale_pos_weight** — multiple approaches compared

## Approach
1. Data cleaning — handled undefined categories in EDUCATION and MARRIAGE
2. Feature engineering — created 4 new features
3. Baseline model — Logistic Regression
4. Advanced models — Random Forest, XGBoost, LightGBM
5. Imbalance handling — SMOTE, class_weight, scale_pos_weight
6. Threshold tuning — PR curve analysis to find optimal classification threshold
7. Model interpretability — XGBoost feature importance

## How to Run Locally

1. Install dependencies:
    pip install streamlit xgboost scikit-learn pandas numpy matplotlib seaborn joblib lightgbm

2. Run the app:
    streamlit run app.py

## Live Demo
[Link to be added after Streamlit Cloud deployment]

## Author
**Prasanna D**  
IIT Gandhinagar (B.Tech, 2022)  
Incoming MS Applied Machine Learning — University of Maryland, Fall 2026