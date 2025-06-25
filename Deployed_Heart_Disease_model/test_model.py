import streamlit as st
import pickle
import numpy as np

# Load preprocessors and model
try:
    with open('le_sex.pkl', 'rb') as f:
        le_sex = pickle.load(f)
    with open('le_chestpain.pkl', 'rb') as f:
        le_chestpain = pickle.load(f)
    with open('le_restingecg.pkl', 'rb') as f:
        le_restingecg = pickle.load(f)
    with open('le_exang.pkl', 'rb') as f:
        le_exang = pickle.load(f)
    with open('le_slope.pkl', 'rb') as f:
        le_slope = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('logreg_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or preprocessors: {e}")
    st.stop()

# Streamlit app
st.title("Heart Disease Prediction App")
st.write("Enter patient details below to predict the risk of heart disease.")

# Organize input fields in two columns
col1, col2 = st.columns(2)

# Numerical inputs
with col1:
    age = st.number_input("Age", min_value=28, max_value=77, value=40, help="Patient's age in years (28–77)")
    resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=200, value=120, help="Resting blood pressure in mmHg")
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=603, value=200, help="Serum cholesterol in mg/dl")
    max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=202, value=120, help="Maximum heart rate achieved (60–202)")
    oldpeak = st.number_input("Oldpeak", min_value=-2.6, max_value=6.2, value=0.0, step=0.1, help="ST depression induced by exercise (–2.6 to 6.2)")

# Categorical inputs (dropdowns)
with col2:
    sex = st.selectbox("Sex", options=["F", "M"], help="Patient's sex (F: Female, M: Male)")
    chest_pain = st.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"], help="Chest pain type: Asymptomatic, Atypical Angina, Non-Anginal Pain, Typical Angina")
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"], help="Fasting blood sugar > 120 mg/dl")
    resting_ecg = st.selectbox("Resting ECG", options=["LVH", "Normal", "ST"], help="Resting electrocardiogram: Left ventricular hypertrophy, Normal, ST-T abnormality")
    exercise_angina = st.selectbox("Exercise-Induced Angina", options=["N", "Y"], help="Exercise-induced angina (N: No, Y: Yes)")
    st_slope = st.selectbox("ST Slope", options=["Down", "Flat", "Up"], help="Slope of the peak exercise ST segment")

# Preprocess inputs
try:
    sex_encoded = le_sex.transform([sex])[0]
    chest_pain_encoded = le_chestpain.transform([chest_pain])[0]
    resting_ecg_encoded = le_restingecg.transform([resting_ecg])[0]
    exercise_angina_encoded = le_exang.transform([exercise_angina])[0]
    st_slope_encoded = le_slope.transform([st_slope])[0]
except ValueError as e:
    st.error(f"Invalid input: {e}. Please check dropdown options.")
    st.stop()
fasting_bs_encoded = 1 if fasting_bs == "Yes" else 0

# Create input array
input_data = np.array([[age, sex_encoded, chest_pain_encoded, resting_bp, cholesterol, fasting_bs_encoded,
                        resting_ecg_encoded, max_hr, exercise_angina_encoded, oldpeak, st_slope_encoded]])

# Scale input
input_data_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict"):
    try:
        prediction = model.predict(input_data_scaled)[0]
        if prediction == 0:
            st.success("**No heart disease risk predicted.**")
        else:
            st.warning("**Potential heart disease risk predicted.**")
    except Exception as e:
        st.error(f"Error during prediction: {e}")