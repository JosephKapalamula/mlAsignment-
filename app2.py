import streamlit as st
st.title("Hello")

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)


# Custom CSS for fonts and colors
custom_css = '''
<style>
body {
font-family: 'Arial', sans-serif;
color: #333;
}

h1, h2, h3, h4, h5, h6 {
color: #4CAF50;
}

.stButton>button {
background-color: #4CAF50;
color: white;
border-radius: 8px;
padding: 10px;
}
</style>
'''

st.markdown(custom_css, unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler,FunctionTransformer

model = joblib.load("best_random_forest_model.pkl")
columns = joblib.load("features.pkl")


# Redefining all the custom functions for preprocessing 
def replace_zeros_with_coumnn_mean(df1):
    columns_to_modify = [col for col in df1.columns if col not in ['Pregnancies', 'Insulin', 'DiabetesPedigreeFunction', 'Outcome']]
    for column in columns_to_modify:
        df1[column].replace(0, df1[column].mean(), inplace=True)
    return df1

def remove_outliers_from_all_columns(df1):
    df2 = df1.copy()
    columns_to_remove_outliers = [col for col in df2.columns if col not in ['Glucose', 'Outcome']]
    
    for column in columns_to_remove_outliers:
        Q1 = df2[column].quantile(0.25)
        Q3 = df2[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df2[column] = df2[column].apply(lambda x: upper_limit if x > upper_limit else (lower_limit if x < lower_limit else x))
        
    return df2
#df3=remove_outliers_from_all_columns(df1)

def scaling_the_numerical_columns(df3):
    scaler = StandardScaler()
    df4 = df3.copy()
    columns_to_transform = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df4[columns_to_transform] = scaler.fit_transform(df4[columns_to_transform])
    return df4

preprocessing_pipeline= joblib.load("preprocessing_pipeline.pkl")


#columns to get inputs from user ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


# tittle of the app
st.title("DIABATES PREDICTING, GROUP 1") 

# input fields for each feature 
Age = st.number_input("Enter your age (required)", min_value=3, max_value=120)

# Check if input is provided
if not Age:  # Assuming age 0 is not a valid input in your case
    st.warning("Please fill out the required field")


Pregnancies= st.number_input("Enter number of preginances,if man enter zero (required)", min_value=0, max_value=14)

# Check if input is provided
if not Pregnancies:
    st.warning("Please fill out the required field")

Glucose= st.number_input("Enter glucose level in milligrams (mg) per deciliter (required)", min_value=1, max_value=1000)

# Check if input is provided
if not Glucose:
    st.warning("Please fill out the required field")

BloodPressure= st.number_input("Enter BloodPressure level in millimeters of mercury (mmHg)", min_value=20, max_value=231)

# Check if input is provided
if not BloodPressure:
    st.warning("Please fill out the required field")

SkinThickness= st.number_input("Enter skin thickness in millimeter(mm)", min_value=0.1 , max_value=5.6)

# Check if input is provided
if not SkinThickness:
    st.warning("Please fill out the required field")

Insulin= st.number_input("Enter level of insulin in µU/mL (microunits per milliliter)", min_value=0.1 , max_value=60.0)

# Check if input is provided
if not Insulin:
    st.warning("Please fill out the required field")

BMI= st.number_input(" Enter Body Mass Index(BMI) in kg/m2 (killogram/hieght square)", min_value=5  ,max_value=55)

# Check if input is provided
if not BMI:
    st.warning("Please fill out the required field")


DiabetesPedigreeFunction= st.number_input("Enter score probability of diabetes based on family history (DiabetesPedigreeFunction) ", min_value=0  ,max_value=4)

# Check if input is provided
if not DiabetesPedigreeFunction:
    st.warning("Please fill out the required field")


if st.button('Predict Diabetes'):
    # Collect input data into a DataFrame
    df1= pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], columns=columns)

    # Scale the input data
    scaled_input = preprocessing_pipeline.transform(df1)

    # Make predictions
    prediction = model.predict(scaled_input)

    # Display the result
    if prediction == 1:
        st.success("The model predicts that person is at risk of diabetes.")
    else:
        st.success("The model predicts that person is not at risk of diabetes.")




