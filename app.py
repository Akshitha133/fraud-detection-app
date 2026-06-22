import streamlit as st
import joblib
import numpy as np

# Load model safely
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detection Portal")
st.markdown("Enter transaction details below to verify potential risk scores.")

# Inputs
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=10.0, step=1.0)
v1_v28_mean = st.slider("Anomalous Signal Index (V-Component Average)", -5.0, 5.0, 0.0)

if st.button("Analyze Transaction"):
    # 1. Create a 30-feature vector
    data = np.zeros(30)
    
    # 2. Fill V1 to V28 with the slider value to simulate dynamic signals
    data[1:29] = v1_v28_mean 
    
    # 3. Assign Amount to the final feature slot
    data[-1] = amount

    # 4. Predict
    prediction = model.predict([data])

    # 5. Display clean feedback
    if prediction[0] == 1:
        st.error("🚨 **High Risk Alert:** This transaction exhibits patterns consistent with fraudulent activity.")
    else:
        st.success("✅ **Transaction Verified:** No immediate anomalies detected.")
