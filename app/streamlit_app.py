import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

# Resolve paths safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "stroke_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

# Load model artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# Title
st.title("🩺 Stroke Risk Prediction System")
st.markdown("Predict stroke risk using patient health and lifestyle data.")
st.markdown("---")

# Input form
st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 1, 100, 45)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])

with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Build input dictionary
input_data = {
    'age': age,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'heart_disease': 1 if heart_disease == "Yes" else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'gender_Male': 1 if gender == "Male" else 0,
    'ever_married_Yes': 1 if ever_married == "Yes" else 0,
    'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
    'work_type_Private': 1 if work_type == "Private" else 0,
    'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
    'work_type_children': 1 if work_type == "children" else 0,
    'Residence_type_Urban': 1 if residence_type == "Urban" else 0,
    'smoking_status_formerly smoked': 1 if smoking_status == "formerly smoked" else 0,
    'smoking_status_never smoked': 1 if smoking_status == "never smoked" else 0,
    'smoking_status_smokes': 1 if smoking_status == "smokes" else 0
}

# Prediction
if st.button("Predict Stroke Risk"):
    input_df = pd.DataFrame([input_data])

    # Align with training columns
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Stroke")
    else:
        st.success(f"✅ Low Risk of Stroke")

    st.write(f"**Predicted Probability of Stroke:** {probability:.2%}")

    # Risk interpretation
    if probability < 0.30:
        st.info("Risk Level: Low")
    elif probability < 0.60:
        st.warning("Risk Level: Moderate")
    else:
        st.error("Risk Level: High")

st.markdown("---")
st.caption("Disclaimer: This app is for educational purposes only and should not be used as medical advice.")