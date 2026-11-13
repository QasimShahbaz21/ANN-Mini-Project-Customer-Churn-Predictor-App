import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model

# Page configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Bank Customer Churn Predictor")
st.write("Predict whether a customer is likely to stay or leave the bank.")

# 1. Load Model and Preprocessor (cached for performance)
@st.cache_resource
def load_artifacts():
    model = load_model("model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
    st.sidebar.success("✅ Model and Preprocessor loaded!")
except Exception as e:
    st.error(f"Error loading model or preprocessor: {e}")
    st.stop()

# 2. Collect Customer Inputs
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    geography = st.selectbox("Country (Geography)", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=18, max_value=100, value=38)
    tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=80000.0, step=1000.0)

# Convert Yes/No back to 1/0
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_member_val = 1 if is_active_member == "Yes" else 0

st.divider()

# 3. Predict on Button Click
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    # Build single-row DataFrame matching the exact original training columns
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card_val,
        'IsActiveMember': is_active_member_val,
        'EstimatedSalary': estimated_salary
    }])

    # Transform using saved preprocessor
    input_processed = preprocessor.transform(input_data)

    # Predict probability with Keras model
    churn_probability = float(model.predict(input_processed, verbose=0)[0][0])
    churn_percent = churn_probability * 100

    # Display results
    st.subheader("Prediction Result")
    st.metric(label="Churn Probability", value=f"{churn_percent:.2f}%")

    if churn_probability >= 0.5:
        st.error(f"⚠️ **Verdict:** Customer is **likely to LEAVE** (Probability: {churn_percent:.1f}%)")
    else:
        st.success(f"✅ **Verdict:** Customer is **likely to STAY** (Probability: {100 - churn_percent:.1f}% retention)")

