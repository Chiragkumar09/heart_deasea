import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


SEX_OPTIONS = {
    "Female": 0,
    "Male": 1,
}

CHEST_PAIN_OPTIONS = {
    "Typical angina": 0,
    "Atypical angina": 1,
    "Non-anginal pain": 2,
    "No chest pain symptoms": 3,
}

YES_NO_OPTIONS = {
    "No": 0,
    "Yes": 1,
}

ECG_OPTIONS = {
    "Normal ECG": 0,
    "ST-T wave abnormality": 1,
    "Possible left ventricular hypertrophy": 2,
}

SLOPE_OPTIONS = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2,
}

THALASSEMIA_OPTIONS = {
    "Unknown / not recorded": 0,
    "Fixed defect": 1,
    "Normal blood flow": 2,
    "Reversible defect": 3,
}

MODEL_PATH = Path(__file__).resolve().parent / "heart_disease_best_model.pickle"


@st.cache_resource
def load_model_bundle():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def convert_user_input_to_model_input(user_input, model_bundle):
    input_data = pd.DataFrame([user_input])

    # Explicitly define all possible categories from your training dataset
    # to prevent pd.get_dummies from dropping them on a single row.
    possible_categories = {
        "cp": [0, 1, 2, 3],
        "restecg": [0, 1, 2],
        "slope": [0, 1, 2],
        "ca": [0, 1, 2, 3, 4],
        "thal": [0, 1, 2, 3]
    }

    # Convert columns to Categorical types with fixed categories
    for col in model_bundle["categorical_columns"]:
        if col in possible_categories:
            input_data[col] = pd.Categorical(input_data[col], categories=possible_categories[col])

    # Now pd.get_dummies will correctly generate the columns and apply drop_first
    input_data_encoded = pd.get_dummies(
        input_data,
        columns=model_bundle["categorical_columns"],
        drop_first=True,
        dtype=int,
    )

    input_data_encoded = input_data_encoded.reindex(
        columns=model_bundle["feature_columns"],
        fill_value=0,
    )

    input_data_scaled = input_data_encoded.copy()
    input_data_scaled[model_bundle["columns_to_scale"]] = model_bundle["scaler"].transform(
        input_data_encoded[model_bundle["columns_to_scale"]]
    )

    return input_data_scaled


def predict_heart_disease(user_input):
    model_bundle = load_model_bundle()
    model_input = convert_user_input_to_model_input(user_input, model_bundle)
    prediction = model_bundle["model"].predict(model_input)[0]

    return {
        "prediction": int(prediction),
    }


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="heart",
    layout="centered",
)

st.title("Heart Disease Prediction")
st.write("Enter patient details in normal medical terms. The app converts them to the model format automatically.")

with st.form("prediction_form"):
    st.subheader("Patient Information")

    age = st.number_input("Age", min_value=1, max_value=120, value=62)
    sex_text = st.selectbox("Sex", options=list(SEX_OPTIONS.keys()))
    chest_pain_text = st.selectbox(
        "Chest Pain Type",
        options=list(CHEST_PAIN_OPTIONS.keys()),
    )
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=140)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=80, max_value=700, value=268)
    fasting_sugar_text = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=list(YES_NO_OPTIONS.keys()),
    )
    ecg_text = st.selectbox(
        "Resting ECG Result",
        options=list(ECG_OPTIONS.keys()),
    )
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=160)
    exercise_angina_text = st.selectbox(
        "Exercise Induced Angina",
        options=list(YES_NO_OPTIONS.keys()),
    )
    oldpeak = st.number_input("ST Depression Value (Oldpeak)", min_value=0.0, max_value=10.0, value=3.6, step=0.1)
    slope_text = st.selectbox(
        "ST Slope During Exercise",
        options=list(SLOPE_OPTIONS.keys()),
    )
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])
    thalassemia_text = st.selectbox(
        "Thalassemia",
        options=list(THALASSEMIA_OPTIONS.keys()),
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    user_input = {
        "age": age,
        "sex": SEX_OPTIONS[sex_text],
        "cp": CHEST_PAIN_OPTIONS[chest_pain_text],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": YES_NO_OPTIONS[fasting_sugar_text],
        "restecg": ECG_OPTIONS[ecg_text],
        "thalach": thalach,
        "exang": YES_NO_OPTIONS[exercise_angina_text],
        "oldpeak": oldpeak,
        "slope": SLOPE_OPTIONS[slope_text],
        "ca": ca,
        "thal": THALASSEMIA_OPTIONS[thalassemia_text],
    }

    result = predict_heart_disease(user_input)

    if result["prediction"] == 0:
        st.success("Prediction: No Heart Disease")
        st.write("The model predicts that this person does not have heart disease.")
    else:
        st.error("Prediction: Heart Disease Detected")
        st.write("The model predicts that this person may have heart disease.")

    st.caption("This is a machine learning prediction and should not replace medical advice.")