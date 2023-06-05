import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the pre-trained model
clf_rf = RandomForestClassifier(random_state=42)

# Load the dataset
data = pd.read_csv("breast_cancer_dataset.csv")
y = data.diagnosis

# Select the relevant columns for prediction
selected_features = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
                     'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

x_selected = data[selected_features]

# Train the model
clf_rf.fit(x_selected, y)

# Create a function to predict diagnosis
def predict_diagnosis(features):
    prediction = clf_rf.predict([features])
    return prediction[0]

# Create a Streamlit app
def main():
    st.title("Breast Cancer Diagnosis")
    st.write("Enter the necessary features to predict the type of diagnosis.")

    # Create input fields for features
    input_features = {}
    for feature in selected_features:
        value = st.number_input(feature)
        input_features[feature] = value

    if st.button("Predict"):
        # Perform prediction
        diagnosis = predict_diagnosis(list(input_features.values()))
        st.write("The predicted diagnosis is:", diagnosis)

if __name__ == "__main__":
    main()
