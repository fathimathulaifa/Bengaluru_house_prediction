import streamlit as st
import pickle
import pandas as pd

# Load the saved pipeline (preprocessor + Random Forest)
model = pickle.load(open("best_model.pkl", "rb"))

# Load dataset for bar chart
df = pd.read_csv("cleaned_data.csv")  # ensure this file is in the same folder

st.title("Bengaluru House Price Prediction")

# User inputs
location = st.selectbox("Select Location", sorted(df['location'].unique()))
bhk = st.number_input("BHK", min_value=1, step=1)
sqft = st.number_input("Total Sqft", min_value=5)
bath = st.number_input("Bathrooms", min_value=1, step=1)
balcony = st.number_input("Balcony Count", min_value=0, step=1)

# Predict button
if st.button("Predict Price"):
    # ✅ Input validation
    if sqft <= 0:
        st.error("Sqft must be greater than zero.")
    elif bhk <= 0:
        st.error("BHK must be at least 1.")
    else:
        # Prepare input row with exact columns pipeline expects
        input_data = pd.DataFrame([[location, bhk, sqft, bath, balcony]],
                                  columns=['location','bhk','total_sqft','bath','balcony'])
        
        # Pipeline handles preprocessing automatically
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Price: ₹{prediction:.2f} lakhs")

        # ✅ Bar chart
        top_locations = df[df['bhk']==bhk].groupby('location')['price'].mean().nlargest(5)
        st.subheader(f"Top 5 Expensive Locations for {bhk} BHK")
        st.bar_chart(top_locations)
