import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Air Quality Prediction")

# Load dataset
df = pd.read_csv("aqi_data.csv")

# Features
X = df[['PM2.5', 'PM10', 'NO2', 'Temperature', 'Humidity']]
y = df['AQI']

# Train model
model = LinearRegression()
model.fit(X, y)

# Inputs
pm25 = st.number_input("PM2.5", value=100.0)
pm10 = st.number_input("PM10", value=150.0)
no2 = st.number_input("NO2", value=40.0)
temp = st.number_input("Temperature", value=30.0)
humidity = st.number_input("Humidity", value=60.0)

# Prediction
if st.button("Predict"):
    sample = pd.DataFrame(
        [[pm25, pm10, no2, temp, humidity]],
        columns=['PM2.5', 'PM10', 'NO2', 'Temperature', 'Humidity']
    )

    predicted_aqi = model.predict(sample)[0]
    green_score = 100 - (predicted_aqi / 5)

    st.success(f"Predicted AQI: {predicted_aqi:.2f}")
    st.success(f"Green Score: {green_score:.2f}")