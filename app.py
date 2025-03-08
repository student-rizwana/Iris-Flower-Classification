import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model.pkl')

# Streamlit App
st.title("🌸 Iris Flower Classification")
st.write("Predict the species of an Iris flower based on its measurements.")

# Input fields for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, step=0.1)

# Prediction
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"The predicted species is: **{species[prediction[0]]}**")

