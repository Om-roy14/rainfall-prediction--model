import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")[1]  # XGBoost from index 1
scaler = joblib.load("scaler (1).pkl")

# Streamlit UI
st.title("🌦️ Rainfall Prediction App")
st.write("Enter the weather details to predict if it will rain or not.")

# Input fields
temperature= st.number_input("Mean Temperature (°C)", value=25.0)
humidity = st.number_input("Humidity (%)", value=70.0)
wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
pressure = st.number_input("Mean Pressure (hPa)", value=1013.0)
cloud= st.number_input("Cloud", value=40)
sunshine = st.number_input("sunshine", value=9.3)
winddirection = st.number_input("winddirection", value=80)
dewpoint = st.number_input("dewpoint)", value=13.1	)
	# day	pressure	temparature	dewpoint	humidity	cloud	rainfall	sunshine	winddirection	windspeed

if st.button("Predict"):

      
        input_data = np.array([[pressure, temperature, dewpoint, humidity, cloud, sunshine, winddirection, wind_speed]])
        scaled_data = scaler.transform(input_data)

       
        prediction = model.predict(scaled_data)


        if prediction[0] == 1:
            st.success("🌧️ Rainfall Expected")
        else:
            st.info("☀️ No Rainfall Expected")

  