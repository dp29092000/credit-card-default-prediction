import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

st.set_page_config(page_title = 'Model Performance', page_icon = "🎯", layout = 'wide')
st.title("🎯 Model Performance")
st.markdown("---")

st.subheader("Model Comparison")

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'RF SMOTE (0.29)', 'RF SMOTE (0.42)', 
              'XGB Tuned (0.54)', 'XGB SMOTE (0.42)', 'LGBM SMOTE (0.38)'],
    'TP': [828, 966, 764, 784, 799, 790],
    'FN': [499, 361, 563, 543, 528, 537],
    'FP': [1399, 1752, 881, 782, 1068, 1046],
    'TN': [3274, 2921, 3792, 3891, 3753, 3627],
    'Recall (%)': [62, 73, 58, 59, 60, 60]
})

st.dataframe(comparison_df, use_container_width=True,hide_index=True)

st.markdown('---')
st.subheader("Final Model - XGBoost Tuned (Threshold 0.54)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Recall", "59%")
col2.metric("Precision", "50%")
col3.metric("ROC-AUC", "0.78")
col4.metric("PR-AUC", "0.56")

st.markdown("---")
st.subheader("Feature Importance - XGBoost Tuned")

import numpy as np

features = ['PAY_1', 'pay_trend', 'PAY_3', 'PAY_2', 'PAY_AMT2', 
            'PAY_5', 'PAY_4', 'LIMIT_BAL', 'average_credit_utilization',
            'PAY_AMT1']

importance_values = [0.334, 0.152, 0.043, 0.038, 0.030, 
                     0.029, 0.028, 0.024, 0.024, 0.023]


st.markdown("---")
st.subheader("Confusion Matrix - XGBoost Tuned (Threshold 0.54)")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(features, importance_values, color='tan')
ax.invert_yaxis()
ax.set_xlabel('Importance Score')
ax.set_title('Top 10 Feature Importance')
st.pyplot(fig)

st.markdown("---")
st.subheader("Confusion Matrix — XGBoost Tuned (Threshold 0.54)")

col1, col2, col3 = st.columns([0.6, 2, 0.6])

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = [[3891, 782], [543, 784]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                ax=ax)
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)