import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature names
model = joblib.load('model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Streamlit app title
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")
st.title("📊 Credit Risk Prediction App")

# Sidebar input
st.sidebar.header("Applicant Details")

# Collect user input
def user_input():
    age = st.sidebar.slider("Age", 18, 75, 30)
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    job = st.sidebar.selectbox("Job Type", [0, 1, 2, 3])
    housing = st.sidebar.selectbox("Housing", ["own", "free", "rent"])
    saving_accounts = st.sidebar.selectbox("Saving Account", ["little", "moderate", "quite rich", "rich", "no_info"])
    checking_account = st.sidebar.selectbox("Checking Account", ["little", "moderate", "rich", "no_info"])
    credit_amount = st.sidebar.number_input("Credit Amount", 100, 100000, 1000)
    duration = st.sidebar.slider("Duration (months)", 4, 72, 12)
    purpose = st.sidebar.selectbox("Purpose", ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation", "retraining", "other"])

    data = {
        'age': age,
        'sex_male': 1 if sex == 'male' else 0,
        'job': job,
        'housing_own': 1 if housing == 'own' else 0,
        'housing_free': 1 if housing == 'free' else 0,
        'saving_accounts_little': 1 if saving_accounts == 'little' else 0,
        'saving_accounts_moderate': 1 if saving_accounts == 'moderate' else 0,
        'saving_accounts_quite rich': 1 if saving_accounts == 'quite rich' else 0,
        'saving_accounts_rich': 1 if saving_accounts == 'rich' else 0,
        'checking_account_little': 1 if checking_account == 'little' else 0,
        'checking_account_moderate': 1 if checking_account == 'moderate' else 0,
        'checking_account_rich': 1 if checking_account == 'rich' else 0,
        'credit_amount': credit_amount,
        'duration': duration,
        'purpose_radio/TV': 1 if purpose == "radio/TV" else 0,
        'purpose_education': 1 if purpose == "education" else 0,
        'purpose_furniture/equipment': 1 if purpose == "furniture/equipment" else 0,
        'purpose_car': 1 if purpose == "car" else 0,
        'purpose_business': 1 if purpose == "business" else 0,
        'purpose_domestic appliances': 1 if purpose == "domestic appliances" else 0,
        'purpose_repairs': 1 if purpose == "repairs" else 0,
        'purpose_vacation': 1 if purpose == "vacation" else 0,
        'purpose_retraining': 1 if purpose == "retraining" else 0,
    }

    return pd.DataFrame([data])

# Get user input
input_df = user_input()

# Fill missing columns if any
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns with 0 value

# Reorder columns
input_df = input_df[feature_names]

# Predict
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)[0]
    result = "Good" if prediction == 1 else "Bad"
    st.subheader(f"Prediction: 🏦 The applicant has a **{result} Credit Risk**.")
