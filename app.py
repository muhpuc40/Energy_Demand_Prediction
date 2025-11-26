import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

st.title("Energy Prediction App")
st.write("Enter the same feature values that were used in model training.")

# Numeric inputs
TotalHours = st.number_input("Total Hours", min_value=0.0, step=0.01)
Bill = st.number_input("Bill", min_value=0.0, step=0.01)
Month = st.number_input("Month", min_value=1, max_value=12)
Week_Number = st.number_input("Week Number", min_value=1, max_value=53)
Day_of_Week = st.number_input("Day of Week (0=Mon .. 6=Sun)", min_value=0, max_value=6)
Day_of_Year = st.number_input("Day of Year", min_value=1, max_value=366)

# Categorical numeric inputs (because df_encoded already converted them)
HUBId = st.number_input("HUBId (numeric value from df_encoded)", min_value=0)
User = st.number_input("User (numeric value from df_encoded)", min_value=0)

Year = st.number_input("Year", min_value=2000, max_value=2100)

if st.button("Predict"):
    try:
        features = np.array([
            TotalHours,
            Bill,
            Month,
            Week_Number,
            Day_of_Week,
            Day_of_Year,
            HUBId,
            User,
            Year
        ]).reshape(1, -1)

        prediction = model.predict(features)[0]
        st.success(f"Predicted Energy: {prediction:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")
