import streamlit as st
import numpy as np
import pandas as pd
import joblib

# === Load model and scaler ===
model_path = r"C:\Users\Manoj\OneDrive\Desktop\DAPF\model\lasso_model.pkl"
scaler_path = r"C:\Users\Manoj\OneDrive\Desktop\DAPF\model\scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("🐐 Goat Farm Net Worth Estimator")
st.markdown("Provide just **three key inputs**. We’ll do the rest internally.")

# === Collect Essential Inputs ===
invested = st.number_input("💰 Invested Amount (INR)", value=15000)
weight_gained = st.number_input("⚖️ Weight Gained (kg)", value=200)
early_weight = st.number_input("🐐 Early Weight (kg)", value=20)

# === Build Full Input with Defaults ===
# Load default values from training data (for simplicity, we hardcode them here)
default_values = {
    'Initial_Strenght': 3,
    'New_Born': 1,
    'Death': 0,
    'Sold': 1,
    'New_Purchase': 1,
    'Current_Strength': 4,
    'Goat_Sell_income_only': 20000
}

# Construct the full feature input
input_dict = {
    'Initial_Strenght': default_values['Initial_Strenght'],
    'New_Born': default_values['New_Born'],
    'Death': default_values['Death'],
    'Sold': default_values['Sold'],
    'New_Purchase': default_values['New_Purchase'],
    'Current_Strength': default_values['Current_Strength'],
    'Early_Weight': early_weight,
    'Weight_Gained': weight_gained,
    'Goat_Sell_income_only': default_values['Goat_Sell_income_only'],
    'invested': invested
}

input_df = pd.DataFrame([input_dict])

# === Predict ===
if st.button("Predict Net Worth"):
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    profit = prediction - invested
    st.success(f"💰 Predicted Net Worth: ₹ {prediction:,.2f}")
    st.info(f"📈 Estimated Profit: ₹ {profit:,.2f}")
