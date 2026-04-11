import streamlit as st
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide"
)
st.title("💳 Credit Card Default Prediction")
st.markdown("---")

st.markdown("""
### About This Project
This project predicts whether a credit card customer will default on their next payment
using machine learning.

### Dataset
- **Source:** UCI Machine Learning Repository
- **Size:** 30,000 customers
- **Features:** Payment history, bill amounts, demographic information

### Model
- **Algorithm:** XGBoost (tuned)
- **Threshold:** 0.54
- **Key Insight:** PAY_1 (most recent payment status) is the strongest predictor at 33% importance

### Navigate
Use the sidebar to explore EDA, Model Performance, and Predictions.
""")

st.markdown("---")
st.caption("Built by Prasanna D | IIT Gandhinagar | University of Maryland (MAML, Fall 2026)")
