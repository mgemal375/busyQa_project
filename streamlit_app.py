import streamlit as st
import numpy as np
import joblib

# Load the trained models
loaded_rf = joblib.load('best_random_forest_model.pkl')
loaded_xgb = joblib.load('best_xgboost_model.pkl')

# Streamlit app title
st.title('Machine Learning Model Predictions')

# Model selection
model_option = st.selectbox('Select the model you want to use:', ['RandomForestClassifier', 'XGBClassifier'])

# Assuming you need inputs for features to make a prediction
# Here you should include input fields corresponding to the features of your dataset
# This is a generic template; you'll need to adjust it based on your specific feature set
feature_1 = st.number_input('Type', format='%f')
feature_2 = st.number_input('Failure Type', format='%f')
# Add more input fields as required for your model

# The 'Predict' button
if st.button('Predict'):
    input_data = np.array([[feature_1, feature_2]])  # Adjust this depending on the number of features
    if model_option == 'RandomForestClassifier':
        prediction = loaded_rf.predict(input_data)
    elif model_option == 'XGBClassifier':
        prediction = loaded_xgb.predict(input_data)

    st.write(f'The predicted output is: {prediction}')

