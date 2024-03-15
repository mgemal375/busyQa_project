
# # Streamlit App

# import streamlit as st
# import numpy as np
# import joblib
# import pandas as pd

# data = pd.read_csv('C:/Users/pcuser/OneDrive - York University/online courses, webinars/BusyQA - DS ML/final/for streamlit.csv')

# # Load the models from disk
# loaded_rf = joblib.load('best_random_forest_model.pkl')
# loaded_xgb = joblib.load('best_xgboost_model.pkl')

# # Make predictions to test if the loaded models are working as expected (use your test data here)
# #predictions_rf = loaded_rf.predict(X_test)
# #predictions_xgb = loaded_xgb.predict(X_test)



# # Streamlit app title
# st.title('Predictive Maintenance Model')

# # Model selection
# model_option = st.selectbox('Select the model you want to use:', ['RandomForestClassifier', 'XGBClassifier'])

# # Assuming you need inputs for features to make a prediction
# # Here you should include input fields corresponding to the features of your dataset
# # This is a generic template; you'll need to adjust it based on your specific feature set
# feature_1 = st.number_input('Rotational speed', format='%f')
# feature_2 = st.number_input('Torque', format='%f')
# feature_3 = st.selectbox('Type', 
#                          ('M', 'L', 'H'))
# # Add more input fields as required for your model

# # The 'Predict' button
# if st.button('Predict'):
#     if model_option == 'RandomForestClassifier':
#         prediction = loaded_rf.predict(data)
#     elif model_option == 'XGBClassifier':
#         prediction = loaded_xgb.predict(data)
    
#     st.write(f'The predicted output is: {prediction}')


# #####################################################
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

# Assuming you've added all required features here
type = st.selectbox('Type', (0,1,2))
air_temp = st.number_input('Air Temperature', min_value=293.0, max_value=298.0, format='%f')
process_temp = st.number_input('Process Temperature', min_value=300.0, max_value=400.0, format='%f')
rotational_speed = st.number_input('Rotational Speed', min_value=1000, max_value=5000)
torque = st.number_input('Torque', min_value=20.0, max_value=100.0, format='%f')
tool_wear = st.number_input('Tool Wear', min_value=0, max_value=200)
failure_type = st.selectbox('Failure Type', (0,1,2,3,4))

# Adjustments for additional features your model might expect
# Add more input features here if your model was trained with more features

# The 'Predict' button
if st.button('Predict'):
    # Ensure this array includes all features in the correct order
    input_data = np.array([[type, air_temp, process_temp, rotational_speed, torque, tool_wear, failure_type]])

    if model_option == 'RandomForestClassifier':
        prediction = loaded_rf.predict(input_data)
    elif model_option == 'XGBClassifier':
        prediction = loaded_xgb.predict(input_data)
    
    st.write(f'The predicted output is: {prediction}')
