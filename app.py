import streamlit as st
import requests

st.title("Customer Churn Prediction")
st.write("Enter a customer's purchase history to predict their likelihood of churning.")

frequency = st.number_input("Frequency (number of orders)", min_value=1, value=5)
monetary = st.number_input("Monetary (total spend, $)", min_value=0.0, value=500.0)
tenure = st.number_input("Tenure (days as a customer)", min_value=0, value=365)

if st.button("Predict Churn"):
    payload = {
        "Frequency": frequency,
        "Monetary": monetary,
        "Tenure": tenure
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        churn_prob = result["churn_probability"]
        prediction = result["churned_prediction"]

        st.subheader("Result")
        if prediction == 1:
            st.error(f"This customer is likely to churn ({churn_prob:.1%} probability)")
        else:
            st.success(f"This customer is unlikely to churn ({churn_prob:.1%} probability)")

        st.progress(churn_prob)
    else:
        st.error(f"Error: {response.json().get('detail', 'Something went wrong')}")
