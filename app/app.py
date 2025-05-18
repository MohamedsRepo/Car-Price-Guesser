import streamlit as st
import joblib
import numpy as np

# Load model, scaler, encoders
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/car_price_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.title("🚗 Car Price Predictor (Modern Version)")
st.write("Enter your car's details below to estimate its selling price:")

# Inputs
year = st.slider("Year", 2000, 2024, 2018)
km = st.number_input("Kilometers Driven", 1000, 300000, 50000)
fuel = st.selectbox("Fuel Type", encoders["fuel"].classes_)
type_ = st.selectbox("Seller Type", encoders["type"].classes_)
transmission = st.selectbox("Transmission", encoders["transmission"].classes_)
owner = st.selectbox("Ownership", encoders["owner"].classes_)
seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7])
brand = st.selectbox("Car Brand", encoders["brand"].classes_)

# Predict
if st.button("Predict Price"):
    # Encode categorical features
    fuel_enc = encoders["fuel"].transform([fuel])[0]
    type_enc = encoders["type"].transform([type_])[0]
    transmission_enc = encoders["transmission"].transform([transmission])[0]
    owner_enc = encoders["owner"].transform([owner])[0]
    brand_enc = encoders["brand"].transform([brand])[0]

    # Scale year and km
    scaled = scaler.transform([[km, year]])
    km_scaled, year_scaled = scaled[0]

    # Combine all features
    user_input = np.array([[year_scaled, km_scaled, fuel_enc, type_enc, transmission_enc, owner_enc, seats, brand_enc]])
    prediction = model.predict(user_input)

    st.success(f"💰 Estimated Price: ₹{prediction[0]:,.0f}")
