import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction App")
st.write(
    "This is an internal tool for the 'Visit with Us' sales team. It predicts "
    "whether a customer is likely to purchase the new Wellness Tourism Package, "
    "so follow-ups can be prioritised for the customers most likely to buy."
)
st.write("Please enter the customer's details below to get a prediction.")

# --- Collect user input (mirrors the features used in training) ---
Age = st.number_input("Age (customer's age)", min_value=18, max_value=100, value=35)

TypeofContact = st.selectbox(
    "Type of Contact (how the customer was contacted)",
    ["Self Enquiry", "Company Invited"],
)

CityTier = st.selectbox(
    "City Tier (1 = most developed, 3 = least developed)", [1, 2, 3]
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes the sales pitch lasted)", min_value=0, max_value=60, value=10
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"],
)

Gender = st.selectbox("Gender", ["Male", "Female"])

NumberOfPersonVisiting = st.number_input(
    "Number of Persons Visiting (total people on the trip)", min_value=1, max_value=10, value=2
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups (by the salesperson after the pitch)", min_value=0, max_value=10, value=3
)

ProductPitched = st.selectbox(
    "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])

MaritalStatus = st.selectbox(
    "Marital Status", ["Single", "Married", "Divorced", "Unmarried"]
)

NumberOfTrips = st.number_input(
    "Number of Trips (average annual number of trips)", min_value=0, max_value=20, value=2
)

Passport = st.selectbox("Holds a Valid Passport?", ["Yes", "No"])

PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0
)

Designation = st.selectbox(
    "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income (gross monthly income)", min_value=0, value=20000
)

# --- Assemble into a single-row dataframe matching the training schema ---
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    result = "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
    st.write(f"Estimated purchase probability: **{probability:.2%}**")
