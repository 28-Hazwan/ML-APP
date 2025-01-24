import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "Model_rf_clf_hyp.pkl"  
model = joblib.load(MODEL_PATH)

expected_features = [
    'Graduate', 'Male', 'Married', 'Self_Employed', 'Total_Income',
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
    'Dependents_0', 'Dependents_1', 'Dependents_2',
    'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban',
    'Some_Other_Feature1', 'Some_Other_Feature2', 'Some_Other_Feature3'
] 

default_data = pd.DataFrame({
    'Graduate': [1, 0, 1, 1, 0],
    'Male': [1, 1, 0, 1, 0],
    'Married': [1, 0, 1, 0, 1],
    'Self_Employed': [0, 1, 1, 0, 0],
    'Total_Income': [6000, 5500, 7000, 8000, 4500],
    'LoanAmount': [200, 150, 250, 300, 100],
    'Loan_Amount_Term': [360, 180, 360, 180, 360],
    'Credit_History': [1, 0, 1, 1, 0],
    'Dependents_0': [0, 1, 0, 1, 0],
    'Dependents_1': [1, 0, 1, 0, 0],
    'Dependents_2': [0, 0, 0, 1, 1],
    'Property_Area_Rural': [0, 1, 0, 0, 1],
    'Property_Area_Semiurban': [1, 0, 0, 1, 0],
    'Property_Area_Urban': [0, 0, 1, 0, 0],
    'Some_Other_Feature1': [10, 15, 12, 10, 13],
    'Some_Other_Feature2': [20, 25, 20, 18, 23],
    'Some_Other_Feature3': [30, 28, 25, 27, 29],
})

st.write("""
# Loan Application Prediction App
This app predicts if the user is **Applicable for a Loan**!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Graduate = st.sidebar.selectbox('Graduate (1 for Yes, 0 for No)', [1, 0])
    Male = st.sidebar.selectbox('Male (1 for Yes, 0 for No)', [1, 0])
    Married = st.sidebar.selectbox('Married (1 for Yes, 0 for No)', [1, 0])
    Self_Employed = st.sidebar.selectbox('Self-Employed (1 for Yes, 0 for No)', [1, 0])
    Total_Income = st.sidebar.text_input('Total Income', '5000.0')
    LoanAmount = st.sidebar.text_input('Loan Amount', '150.0')
    Loan_Amount_Term = st.sidebar.text_input('Loan Amount Term (in months)', '360.0')
    Credit_History = st.sidebar.selectbox('Credit History (1 for Yes, 0 for No)', [1, 0])
    Dependents_0 = st.sidebar.selectbox('Dependents 0 (1 for Yes, 0 for No)', [1, 0])
    Dependents_1 = st.sidebar.selectbox('Dependents 1 (1 for Yes, 0 for No)', [1, 0])
    Dependents_2 = st.sidebar.selectbox('Dependents 2 (1 for Yes, 0 for No)', [1, 0])
    Property_Area_Rural = st.sidebar.selectbox('Property Area Rural (1 for Yes, 0 for No)', [1, 0])
    Property_Area_Semiurban = st.sidebar.selectbox('Property Area Semiurban (1 for Yes, 0 for No)', [1, 0])
    Property_Area_Urban = st.sidebar.selectbox('Property Area Urban (1 for Yes, 0 for No)', [1, 0])

    data = {
        'Graduate': Graduate,
        'Male': Male,
        'Married': Married,
        'Self_Employed': Self_Employed,
        'Total_Income': float(Total_Income),
        'LoanAmount': float(LoanAmount),
        'Loan_Amount_Term': float(Loan_Amount_Term),
        'Credit_History': Credit_History,
        'Dependents_0': Dependents_0,
        'Dependents_1': Dependents_1,
        'Dependents_2': Dependents_2,
        'Property_Area_Rural': Property_Area_Rural,
        'Property_Area_Semiurban': Property_Area_Semiurban,
        'Property_Area_Urban': Property_Area_Urban
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

df = df.reindex(columns=expected_features, fill_value=0)

prediction = model.predict(df.to_numpy())
prediction_proba = model.predict_proba(df)

st.subheader('Prediction')
st.write("Approved" if prediction[0] else "Rejected")

st.subheader('Prediction Probability')
st.write(f"Approval Probability: {prediction_proba[0][1]:.2f}")
st.write(f"Rejection Probability: {prediction_proba[0][0]:.2f}")

st.subheader('Descriptive Statistics of Default Data')
st.write(default_data.describe())

st.subheader('Distribution of Total Income')
plt.figure(figsize=(8, 6))
sns.histplot(default_data['Total_Income'], kde=True, color='blue')
st.pyplot(plt)



correlation_matrix = default_data.corr()

st.subheader('Correlation Heatmap of Numerical Features')
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(plt)
