import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
st.title("🔮 Default Risk Predictor")
st.markdown("---")

# Load model and dataset
model = joblib.load('xgb_fine_tuned.pkl')
df = pd.read_csv('UCI_Credit_Card.csv')

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    pay_1 = st.selectbox("PAY_1 (Most Recent Payment Status)", 
                         options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                         help="-1=Paid on time, 1=1 month delay, 2=2 month delay...")
    pay_2 = st.selectbox("PAY_2 (2nd Month Payment Status)", 
                         options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_3 = st.selectbox("PAY_3 (3rd Month Payment Status)", 
                         options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_trend = st.slider("Pay Trend (Sum of PAY_1 to PAY_6)", 
                          min_value=-12, max_value=48, value=0)
    
    with col2:
        limit_bal = st.number_input("Credit Limit (NT Dollar)", 
                                    min_value=10000, max_value=1000000, value=100000, step=10000)
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        bill_amt1 = st.number_input("Most Recent Bill Amount (NT Dollar)", 
                                    min_value=0, max_value=500000, value=10000, step=1000)
        pay_amt1 = st.number_input("Most Recent Payment Amount (NT Dollar)", 
                                min_value=0, max_value=500000, value=5000, step=1000)

st.markdown("---")

if st.button("Predict Default Risk", type="primary"):
    
    # Start with median values for all features
    medians = pd.read_csv('median_values.csv', index_col=0).squeeze()
    input_data = medians.to_dict()
    
    # Override with user inputs
    input_data['PAY_1'] = pay_1
    input_data['PAY_2'] = pay_2
    input_data['PAY_3'] = pay_3
    input_data['LIMIT_BAL'] = limit_bal
    input_data['AGE'] = age
    input_data['BILL_AMT1'] = bill_amt1
    input_data['PAY_AMT1'] = pay_amt1
    input_data['Pay_Trend'] = pay_trend
    
    # Convert to dataframe
    input_df = pd.DataFrame([input_data])
    
    # Predict
    prob = model.predict_proba(input_df)[:,1][0]
    prediction = int(prob >= 0.54)

    st.markdown("---")
    
    if prediction == 1:
        st.error(f"⚠️ High Default Risk — Probability: {prob:.1%}")
    else:
        st.success(f"✅ Low Default Risk — Probability: {prob:.1%}")