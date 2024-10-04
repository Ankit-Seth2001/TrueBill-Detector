import streamlit as st
import pandas as pd
import pickle

# Load the model
with open("TrueBill.pickle", "rb") as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Fake or Genuine Predictor")

# Input features
st.header("Enter the features")
variance = st.number_input("Variance", value=0.0)
skewness = st.number_input("Skewness", value=0.0)
curtosis = st.number_input("Curtosis", value=0.0)
entropy = st.number_input("Entropy", value=0.0)

# Button to predict
if st.button("Predict"):
    # Prepare the input data for prediction
    input_data = pd.DataFrame([[variance, skewness, curtosis, entropy]], 
                               columns=["Variance", "Skewness", "Curtosis", "Entropy"])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the result
    if prediction[0] == 0:
        st.success("The input is Fake.")
    else:
        st.success("The input is Genuine.")
