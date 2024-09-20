import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load Dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv'
    return pd.read_csv(url)

# Function to train the model
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    return model, X_train.columns

# Function for making predictions
def make_prediction(model, input_data):
    return model.predict(input_data)

# Main Streamlit app
def main():
    st.title("Heart Disease Prediction")

    # Load and display the dataset
    df = load_data()

    # Standardize continuous features
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    # Train the logistic regression model
    model, feature_names = train_model(df)

    # Create input fields for user input
    st.subheader("Enter Patient Information")
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: f"Type {x+1}")
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    restecg = st.selectbox("Resting ECG Result", [0, 1, 2], format_func=lambda x: f"Result {x}")
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], format_func=lambda x: f"Slope {x+1}")
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # Create input DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Standardize continuous input features
    input_data[continuous_features] = scaler.transform(input_data[continuous_features])

    # Make predictions
    if st.button("Predict"):
        prediction = make_prediction(model, input_data)
        if prediction[0] == 1:
            st.write("The model predicts that the person **has heart disease**.")
        else:
            st.write("The model predicts that the person **does not have heart disease**.")

if __name__ == '__main__':
    main()
