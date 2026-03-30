import streamlit as st
import joblib
import numpy as np

# load model
model = joblib.load("model.pkl")

st.title("💳 Credit Card Fraud Detection")

amount = st.number_input("Transaction Amount", min_value=0.0)

if st.button("Predict"):
    data = np.zeros(30)
    data[-1] = amount

    prediction = model.predict([data])

    if prediction[0] == 1:
        st.error("🚨 Fraud Detected")
    else:
        st.success("✅ Legitimate Transaction")
