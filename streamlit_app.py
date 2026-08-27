import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Customer Churn Prediction")
st.write("Enter a customer's purchase history to predict their likelihood of churning.")

@st.cache_resource
def load_and_train_model():
    data = pd.read_csv('cleaned_retail.csv')
    rfm = pd.read_csv('rfm_table.csv')

    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    rfm['Churned'] = (rfm['Recency'] > 90).astype(int)

    tenure = data.groupby('Customer ID')['InvoiceDate'].agg(['min', 'max'])
    tenure['Tenure'] = (tenure['max'] - tenure['min']).dt.days
    tenure = tenure[['Tenure']].reset_index()
    rfm = rfm.merge(tenure, on='Customer ID', how='left')

    rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    rfm['Freq_Monetary_Interaction'] = rfm['Frequency'] * rfm['Monetary']

    feature_cols = ['Frequency', 'Monetary', 'Tenure', 'AvgOrderValue', 'Frequency_log', 'Monetary_log', 'Freq_Monetary_Interaction']
    X = rfm[feature_cols]
    y = rfm['Churned']

    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X, y)
    return model

model = load_and_train_model()

frequency = st.number_input("Frequency (number of orders)", min_value=1, value=5)
monetary = st.number_input("Monetary (total spend, $)", min_value=0.0, value=500.0)
tenure = st.number_input("Tenure (days as a customer)", min_value=0, value=365)

if st.button("Predict Churn"):
    avg_order_value = monetary / frequency
    frequency_log = np.log1p(frequency)
    monetary_log = np.log1p(monetary)
    interaction = frequency * monetary

    features = pd.DataFrame([{
        'Frequency': frequency,
        'Monetary': monetary,
        'Tenure': tenure,
        'AvgOrderValue': avg_order_value,
        'Frequency_log': frequency_log,
        'Monetary_log': monetary_log,
        'Freq_Monetary_Interaction': interaction
    }])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"This customer is likely to churn ({probability:.1%} probability)")
    else:
        st.success(f"This customer is unlikely to churn ({probability:.1%} probability)")

    st.progress(probability)
