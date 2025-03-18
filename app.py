import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Load the pre-trained model and scalers
model = tf.keras.models.load_model("model.h5")

# Load the encoders and scalers
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder-geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")


# User input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, format="%.2f")
num_of_products = st.slider("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary")
credit_score = st.number_input("Credit Score")

# Create input data dictionary
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],  # Encode gender
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)

# One-hot encode the geographical data safely
try:
    geo_encoded = onehot_encoder_geo.transform(np.array([[geography]]))  # FIXED
    geo_encoded_df = pd.DataFrame(
        geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )
except Exception as e:
    st.error(f"Error encoding geography: {e}")
    st.stop()

# Concatenate the data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display the result
st.write(f"**Churn Probability:** {prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.error("🚨 The customer is likely to churn!")
else:
    st.success("✅ The customer is likely to stay.")
