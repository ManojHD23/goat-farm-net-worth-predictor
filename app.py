import streamlit as st
import pandas as pd
import joblib

# ==== Load Model and Scaler ====
model = joblib.load("lasso_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==== Streamlit Page Setup ====
st.set_page_config(page_title="🐐 Goat Farm Net Worth Predictor", layout="centered")
st.title("🐐 Goat Farm Net Worth Predictor")
st.markdown("### Enter only 3 values to estimate your goat farm's net worth.")

# ==== Input Form ====
with st.form("prediction_form"):
    invested = st.number_input("💰 Invested Amount (INR)", step=1, format="%d")
    early_weight = st.number_input("⚖️ Early Weight (kg)", step=1, format="%d")
    weight_gained = st.number_input("💪 Weight Gained (kg)", step=1, format="%d")
    
    submitted = st.form_submit_button("🚀 Predict Net Worth")

# ==== Prediction Logic ====
if submitted:
    # Feature columns used during training
    feature_columns = [
        'Initial_Strenght', 'New_Born', 'Death', 'Sold', 'New_Purchase',
        'Current_Strength', 'Early_Weight', 'Weight_Gained', 'Goat_Sell_income_only', 'invested'
    ]

    # Default values for all other features (fixed)
    default_values = {
        'Initial_Strenght': 3,
        'New_Born': 1,
        'Death': 0,
        'Sold': 1,
        'New_Purchase': 1,
        'Current_Strength': 4,
        'Goat_Sell_income_only': 20000
    }

    # Inject client input into default dict
    default_values['Early_Weight'] = int(early_weight)
    default_values['Weight_Gained'] = int(weight_gained)
    default_values['invested'] = int(invested)

    # Create input DataFrame with correct order
    input_df = pd.DataFrame([[default_values[col] for col in feature_columns]], columns=feature_columns)

    # Transform and predict
    X_scaled = scaler.transform(input_df)
    predicted_net_worth = model.predict(X_scaled)[0]
    predicted_net_worth = round(predicted_net_worth, 2)

    # ROI
    profit = round(predicted_net_worth - default_values['invested'], 2)

    # ==== Display Output ====
    st.subheader("📈 Prediction Result")
    st.success(f"💵 Predicted Net Worth: ₹ {predicted_net_worth:,.2f}")
    st.info(f"📊 Estimated Profit: ₹ {profit:,.2f}")
    if profit > 0:
        st.balloons()
