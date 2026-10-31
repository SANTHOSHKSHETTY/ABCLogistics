import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title='Delivery Delay Predictor', layout='centered')

st.title('📦 Delivery Delay Prediction App')
st.write('Enter the details of the shipment below to predict the probability of a delivery delay.')

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('logistic_regression_model.pkl')

model = load_model()

# Create input form for user features
with st.form('prediction_form'):
    st.subheader('Delivery Attributes')
    
    # Columns layout for a cleaner interface
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_distance = st.number_input('Delivery Distance (miles)', min_value=0.0, value=25.0, step=0.1)
        traffic_congestion = st.slider('Traffic Congestion Score (1-5)', min_value=1, max_value=5, value=3)
        weather_condition = st.slider('Weather Condition Severity (1-3)', min_value=1, max_value=3, value=2)
        delivery_slot = st.slider('Delivery Slot (1-3)', min_value=1, max_value=3, value=2)
        driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=10, step=1)
        num_stops = st.number_input('Number of Stops', min_value=1, value=5, step=1)
        
    with col2:
        vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=5, step=1)
        road_condition = st.slider('Road Condition Score (1-5)', min_value=1, max_value=5, value=3)
        package_weight = st.number_input('Package Weight (lbs)', min_value=0.0, value=25.0, step=0.1)
        fuel_efficiency = st.number_input('Fuel Efficiency (mpg)', min_value=0.0, value=12.0, step=0.1)
        warehouse_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60, step=1)
        
    submit = st.form_submit_button('Predict Delay')

if submit:
    # Map inputs to DataFrame schema
    input_data = pd.DataFrame([{
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_time
    }])
    
    # Run prediction
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]
    
    st.markdown('---')
    st.subheader('Prediction Result')
    if prediction == 1:
        st.error(f'⚠️ **Delayed**: There is a **{probability*100:.2f}%** chance that this delivery will be delayed.')
    else:
        st.success(f'✅ **On Time**: There is a **{(1-probability)*100:.2f}%** chance that this delivery will arrive on time.')
