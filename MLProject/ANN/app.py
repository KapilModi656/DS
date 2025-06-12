# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import os
st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir('.'))
# Load model and preprocessor
model = tf.keras.models.load_model('churn_model.h5',compile=False)  # Make sure this exists
preprocessor = joblib.load('preprocessor.pkl')  # Make sure this exists

st.title("🔍 Customer Churn Prediction App")
st.write("Enter customer data below to predict if they will churn (exit) or not.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
age = st.number_input("Age", 18, 100, 35)
credit_score = st.number_input("Credit Score", 300, 900, 600)
tenure = st.slider("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 1000, 300000, 50000)
if(gender=='Male'):
    gender=1
else:
    gender=0
# Collect data
input_dict = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active],
    'EstimatedSalary': [estimated_salary]
}
input_df = pd.DataFrame(input_dict)

# Predict
if st.button("Predict"):
    try:
        input_transformed = preprocessor.transform(input_df)
        pred = model.predict(input_transformed)
        label = "❌ Customer Will Exit" if pred[0][0] > 0.5 else "✅ Customer Will Stay"
        st.success(f"Prediction: {label}")
    except Exception as e:
        st.error(f"Error: {e}")
