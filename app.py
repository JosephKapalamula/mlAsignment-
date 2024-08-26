import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from PIL import Image
import cv2
from io import BytesIO
import base64


# Load and prepare the second image (for the left side)
image2 = cv2.imread("staticimages/dibates.jpg")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2 = Image.fromarray(image2)

# Encode the background image to base64
def get_base64_image(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

background_image = get_base64_image(image2)

# Set the background image using custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{background_image}");
        background-size: 100%; /* Increase the background size */
        background-position: center; /* Center the background image */
    }}
    </style>
    """,
    unsafe_allow_html=True
)
custom_css = '''
<style>
.stApp {
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
    border: none;
    cursor: pointer;
}

.stButton>button:hover {
    background-color: #45a049;
}
</style>
'''

st.markdown(custom_css, unsafe_allow_html=True)
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
custom_css = """
<style>
    .title-text {
        color: black; /* Black color for title */
    }

    .custom-label {
        color: black; /* Black color for input labels */
        font-weight: bold; /* Bold text for labels */
        margin-bottom: -1px; /* Adjusts spacing between label and input field */
    }

    .stNumberInput input {
        background-color: white; /* White background for input fields */
        color: black; /* Black text color for input fields */
        font-weight: bold; /* Bold text for input fields */
        border: 1px solid #ccc;
        padding: 5px;
        border-radius: 5px;

        
    }
    .stNumberInput button {
        display: none;
    }
    .success-container {
    background-color: blue;
    padding: 10px;
    border: 1px solid #ccc;
    width=500px
}

.success-container p {
    color: black;
}
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Title with custom color
st.markdown('<h1 class="title-text">DIABETES PREDICTING, GROUP 1</h1>', unsafe_allow_html=True)

# Use st.ma
st.markdown('<div class="custom-label">Enter your age (required)</div>', unsafe_allow_html=True)
Age = st.number_input(label='a',min_value=3, max_value=120, value=None,label_visibility='hidden')

st.markdown('<div class="custom-label">Enter number of pregnancies, if man enter zero (required)</div>', unsafe_allow_html=True)
Pregnancies = st.number_input(label='b', min_value=0, max_value=14, value=None,label_visibility='hidden')

st.markdown('<div class="custom-label">Enter glucose level in milligrams (mg) per deciliter (required)</div>', unsafe_allow_html=True)
Glucose = st.number_input('hy', min_value=1, max_value=1000, value=None,label_visibility='hidden')

st.markdown('<div class="custom-label">Enter BloodPressure level in millimeters of mercury (mmHg)</div>', unsafe_allow_html=True)
BloodPressure = st.number_input('hc', min_value=20, max_value=231, value=None,label_visibility='hidden')

st.markdown('<div class="custom-label">Enter skin thickness in millimeters (mm)</div>', unsafe_allow_html=True)
SkinThickness = st.number_input('md', min_value=0.1, max_value=5.6, value=None,label_visibility='hidden')

st.markdown('<div class="custom-label">Enter level of insulin in µU/mL (microunits per milliliter)</div>', unsafe_allow_html=True)
Insulin = st.number_input('rr', min_value=0.1, max_value=60.0, value=None,label_visibility='hidden')

st.markdown('<div class="custom-label">Enter Body Mass Index (BMI) in kg/m2 (kilogram/height square)</div>', unsafe_allow_html=True)
BMI = st.number_input('mr', min_value=5, max_value=55, value=None,label_visibility='hidden')

st.markdown('<div class="custom-label">Enter score probability of diabetes based on family history (DiabetesPedigreeFunction)</div>', unsafe_allow_html=True)
DiabetesPedigreeFunction = st.number_input('mm', min_value=0, max_value=4, value=None,label_visibility='hidden')

# Check if all required fields are filled before allowing prediction
if None in [Age, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction]:
    st.markdown('<div class="custom-label" style="color: #FF0033;">Please fill out all required fields.</div>', unsafe_allow_html=True)
else:
    if st.button('Predict Diabetes'):
        # Collect input data into a DataFrame (assuming 'columns' and 'preprocessing_pipeline' are defined)
        df1 = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], columns=columns)
        
        # Scale the input data
        #scaled_input = preprocessing_pipeline.transform(df1)

        # Make predictions
        prediction = model.predict(df1)

        # Display the result h
        if prediction == 1:
            st.markdown('<div class="success-container"><p>The model predicts that the person is at risk of diabetes!</p></div>', unsafe_allow_html=True)
        else:
            st.success('<div class="success-container"><p>The model predicts that the person is not at risk of diabetes.</p></div>', unsafe_allow_html=True)
