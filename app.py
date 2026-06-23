import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model
model = load_model("diabetes_model.keras")

# Load scaler
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient details below:")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):

    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Apply scaling
    data = scaler.transform(data)

    prediction = model.predict(data)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        st.error(f"Diabetic (Probability: {probability:.2%})")
    else:
        st.success(f"Non-Diabetic (Probability: {probability:.2%})")