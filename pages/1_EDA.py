import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.markdown("---")

df = pd.read_csv('UCI_Credit_Card.csv')

st.subheader("Target Variable Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x='default.payment.next.month', data=df, 
              hue='default.payment.next.month', legend=False, 
              palette='husl', ax=ax)
ax.set_title('Target Variable distribution')
ax.set_xlabel('No Default(0) vs Default(1)')
ax.set_ylabel('Count')

for p in ax.patches:
  ax.annotate(f'{p.get_height():.0f}',
  (p.get_x() + p.get_width()/2, p.get_height()),
  va = 'bottom', ha = 'center')

st.pyplot(fig)

st.markdown("---")
st.subheader("Default Rate by PAY_1 Status")

fig, ax = plt.subplots(figsize=(10, 5))
pay1_default = df.groupby('PAY_0')['default.payment.next.month'].mean() * 100
pay1_default.plot(kind='bar', color='coral', edgecolor='white', ax=ax)
ax.set_title('Default Rate by PAY_1 Status')
ax.set_xlabel('PAY_1 Status')
ax.set_ylabel('Default Rate %')
ax.tick_params(axis='x', rotation=0)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=9)
st.pyplot(fig)


st.markdown("---")
st.subheader("Default Rate by Categorical Variables")

fig, axes = plt.subplots(3, 1, figsize=(10, 15))
cat_features = ['SEX', 'EDUCATION', 'MARRIAGE']

for i, col in enumerate(cat_features):
    default_rate = df.groupby(col)['default.payment.next.month'].mean() * 100
    ecolor = ['darkgray','orchid','royalblue']
    default_rate.plot(kind='bar', ax=axes[i], color=ecolor[i], edgecolor='white')
    axes[i].set_title(f'Default Rate by {col}')
    axes[i].set_ylabel('Default Rate %')
    axes[i].set_xlabel(col)
    axes[i].tick_params(axis='x', rotation=0)

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.subheader("Correlation Heatmap")

fig, ax = plt.subplots(figsize=(15, 10))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), 
            annot=False,
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            ax=ax)
ax.set_title('Correlation Heatmap')
st.pyplot(fig)