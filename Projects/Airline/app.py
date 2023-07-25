import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("SVC.pkl")


def get_user_input():
    # Inflight wifi service
    wifi_service = st.number_input("Inflight wifi service", min_value=0.0, max_value=5.0, value=3.0)
    # Ease of Online booking
    online_booking = st.number_input("Ease of Online booking", min_value=0.0, max_value=5.0, value=3.0)
    # Online boarding
    online_boarding = st.number_input("Online boarding", min_value=0.0, max_value=5.0, value=3.0)
    # Seat comfort
    seat_comfort = st.number_input("Seat comfort", min_value=0.0, max_value=5.0, value=3.0)
    # Inflight entertainment
    inflight_entertainment = st.number_input("Inflight entertainment", min_value=0.0, max_value=5.0, value=3.0)
    # On-board service
    onboard_service = st.number_input("On-board service", min_value=0.0, max_value=5.0, value=3.0)
    # Leg room service
    leg_room_service = st.number_input("Leg room service", min_value=0.0, max_value=5.0, value=3.0)
    # Type of Travel (Business or Personal)
    type_of_travel = st.radio("Type of Travel", ["Business", "Personal"])
    # Class (First, Business, Economy)
    class_type = st.radio("Class", ["Business", "Eco Plus", "Eco"])

    # Encode Type of Travel and Class
    type_of_travel_encoded = 1.0 if type_of_travel == "Business" else 0.0
    class_encoded = 1.0 if class_type == "Eco" else 2.0 if class_type == "Eco Plus" else 3.0

    # Convert input into a DataFrame
    data = {
        "Inflight wifi service": [wifi_service],
        "Ease of Online booking": [online_booking],
        "Online boarding": [online_boarding],
        "Seat comfort": [seat_comfort],
        "Inflight entertainment": [inflight_entertainment],
        "On-board service": [onboard_service],
        "Leg room service": [leg_room_service],
        "Type of Travel_Business travel": [type_of_travel_encoded],
        "Type of Travel_Personal Travel": [1.0 - type_of_travel_encoded],
        "Class_Encoded": [class_encoded]
    }
    user_input_df = pd.DataFrame(data)

    return user_input_df

    # Convert input into a DataFrame
    data = {
        "Inflight wifi service": [wifi_service],
        "Ease of Online booking": [online_booking],
        "Online boarding": [online_boarding],
        "Seat comfort": [seat_comfort],
        "Inflight entertainment": [inflight_entertainment],
        "On-board service": [onboard_service],
        "Leg room service": [leg_room_service],
        "Type of Travel_Business travel": [type_of_travel_encoded],
        "Type of Travel_Personal Travel": [1.0 - type_of_travel_encoded],
        "Class_Encoded": [class_encoded]
    }
    user_input_df = pd.DataFrame(data)

    return user_input_df

def predict_output(user_input_df):
    # Scale user input using the loaded scaler
    scaled_input = scaler.transform(user_input_df)

    # Make predictions using the model
    predictions = model.predict(scaled_input)

    return predictions

def main():
    # Title and description
    st.title("Airline Satisfaction Prediction")
    st.write("Please provide your feedback for each feature:")

    # Get user input
    user_input_df = get_user_input()

    # Display user input as a table
    st.subheader("Your Input:")
    st.dataframe(user_input_df)

    
    # Add a "Predict" button
    if st.button("Predict"):
        # Predict output
        prediction = predict_output(user_input_df)

        # Display prediction
        st.subheader("Prediction:")
        st.write(prediction)
   
if __name__ == "__main__":
    main()
