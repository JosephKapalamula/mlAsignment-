from sklearn.pipeline import Pipeline

# Redefine the custom functions
def replace_zeros_with_column_mean(df1):
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

def scaling_the_numerical_columns(df3):
    scaler = StandardScaler()
    df4 = df3.copy()
    columns_to_transform = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df4[columns_to_transform] = scaler.fit_transform(df4[columns_to_transform])
    return df4

# Load the saved pipeline
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")