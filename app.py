import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# --- Load Model and Scaler ---
# Load the pre-trained RandomForest model
try:
    with open('models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("The 'model.pkl' file was not found. Please run the training script first to generate it.")
    st.stop()

# Load the fitted scaler object
try:
    with open('models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("The 'scaler.pkl' file was not found. Please run the training script first to generate it.")
    st.stop()

# --- Streamlit App Interface ---
st.title('ISP Customer Churn Prediction')
st.write('Please enter the customer details below and click "Predict".')

# --- User Input for Prediction ---
st.header('Customer Input Features')

# Create columns for a more organized layout
col1, col2 = st.columns(2)

with col1:
    is_tv_subscriber = st.selectbox('Is TV Subscriber?', (0, 1), help="0 = No, 1 = Yes")
    subscription_age = st.slider('Subscription Age (years)', 0.0, 12.0, 5.0)
    reamining_contract = st.slider('Remaining Contract (years)', 0.0, 3.0, 1.0)
    upload_avg = st.slider('Average Upload (GB)', 0.0, 20.0, 5.0)

with col2:
    is_movie_package_subscriber = st.selectbox('Is Movie Package Subscriber?', (0, 1), help="0 = No, 1 = Yes")
    bill_avg = st.slider('Average Monthly Bill ($)', 0, 100, 25)
    download_avg = st.slider('Average Download (GB)', 0.0, 100.0, 50.0)

# Create a DataFrame from the user inputs
data = {
    'is_tv_subscriber': is_tv_subscriber,
    'is_movie_package_subscriber': is_movie_package_subscriber,
    'subscription_age': subscription_age,
    'bill_avg': bill_avg,
    'reamining_contract': reamining_contract,
    'download_avg': download_avg,
    'upload_avg': upload_avg,
}
input_df = pd.DataFrame(data, index=[0])


# Prediction button
if st.button('Predict'):
    # Ensure the column order matches the model's training data (with underscores)
    expected_columns = ['is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age', 'bill_avg', 
                        'reamining_contract', 'download_avg', 'upload_avg']
    input_df_ordered = input_df[expected_columns]
    
    # Scale the user input using the loaded scaler
    input_scaled = scaler.transform(input_df_ordered)

    # Make prediction using the loaded model
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction Result')
    churn_status = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f'**Will the customer churn?** `{churn_status}`')

    st.subheader('Prediction Probability')
    st.write(f'**Probability of Not Churning:** `{prediction_proba[0][0]:.2f}`')
    st.write(f'**Probability of Churning:** `{prediction_proba[0][1]:.2f}`')
