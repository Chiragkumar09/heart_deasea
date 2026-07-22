

# Heart Disease Prediction

## About the Project

This project predicts whether a person has heart disease using machine learning.

The model takes patient medical information as input and predicts:

- 0 → No Heart Disease
- 1 → Heart Disease

---

## Dataset

The dataset contains **303 patient records** with **13 medical features**.

Some important features are:

- Age
- Gender
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Maximum Heart Rate
- Exercise Induced Angina
- Number of Major Blood Vessels
- Thalassemia

Target:

- 0 = No Heart Disease
- 1 = Heart Disease

---

## Project Steps

- Load the dataset
- Clean and preprocess the data
- Scale numerical features using StandardScaler
- Train the machine learning model
- Evaluate the model
- Save the trained model and scaler
- Build a Streamlit web application for prediction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---

## Project Files

```
Heart_Disease_Prediction/
│
├── app.py
├── train.py
├── evaluation.py
├── data_processing.py
├── model.pkl
├── scaler.pkl
├── heart_disease_data.csv
└── README.md
```

---

## How to Run

Train the model:

```bash
python train.py
```

Evaluate the model:

```bash
python evaluation.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```