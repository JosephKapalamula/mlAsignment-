import streamlit as st 
Age = st.number_input("Enter your age (required)", min_value=3, max_value=120, value=None)
Pregnancies = st.number_input("Enter number of pregnancies, if man enter zero (required)", min_value=0, max_value=14, value=None)
Glucose = st.number_input("Enter glucose level in milligrams (mg) per deciliter (required)", min_value=1, max_value=1000, value=None)
BloodPressure = st.number_input("Enter BloodPressure level in millimeters of mercury (mmHg)", min_value=20, max_value=231, value=None)
SkinThickness = st.number_input("Enter skin thickness in millimeter(mm)", min_value=0.1, max_value=5.6, value=None)
Insulin = st.number_input("Enter level of insulin in µU/mL (microunits per milliliter)", min_value=0.1, max_value=60.0, value=None)
BMI = st.number_input("Enter Body Mass Index(BMI) in kg/m2 (killogram/hieght square)", min_value=5, max_value=55, value=None)
DiabetesPedigreeFunction = st.number_input("Enter score probability of diabetes based on family history (DiabetesPedigreeFunction)", min_value=0, max_value=4, value=None)

# Check if all required fields are filled before allowing prediction
if None in [Age, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction]:
    st.warning("Please fill out all required fields.")