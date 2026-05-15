#  Stroke Risk Prediction System

An end-to-end machine learning project that predicts stroke risk using patient health and lifestyle data.

##  Project Overview
This project uses machine learning to predict whether a person is at risk of stroke based on features such as age, hypertension, heart disease, glucose level, BMI, smoking status, and more.

The goal of this project is to:
- perform exploratory data analysis
- preprocess healthcare data
- handle class imbalance using SMOTE
- train and compare multiple classification models
- optimize for **recall**, since missing a stroke case is costly
- deploy the final model as a Streamlit web app

---

##  Project Structure

```text
stroke-risk-prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── healthcare-dataset-stroke-data.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── models/
│   ├── scaler.pkl
│   ├── stroke_model.pkl
│   └── feature_names.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── reports/
│   └── figures/
│
├── README.md
├── requirements.txt
└── .gitignore

## Live Demo
Deployed Streamlit App: https://stroke-risk-prediction-pdil8wz7gbcuo9chapumxd.streamlit.app/)