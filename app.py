
import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from datetime import time

# Load the trained model and vectorizer
log_reg_model = joblib.load('log_reg_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to preprocess input and make predictions
def predict_sentiment(review, hour, rating):
    # Clean and preprocess the review text
    review_cleaned = review.lower()
    # Vectorize the review text
    review_tfidf = tfidf_vectorizer.transform([review_cleaned])
    # Create numeric features (hour and rating)
    numeric_features = np.array([[hour, rating]])
    # Combine TF-IDF and numeric features
    features = hstack([review_tfidf, numeric_features])
    # Make prediction
    prediction = log_reg_model.predict(features)
    return prediction[0]

# Streamlit app
st.title("TikTok Review Sentiment Analysis")

# Input bar for text
review = st.text_input("Enter your review:")

# Time picker widget for selecting the hour
selected_time = st.time_input("Select the time of the day:", value=time(12, 0))  # Default to 12:00 PM
hour = selected_time.hour  # Extract the hour from the selected time

# Slider for rating (1-5)
rating = st.slider("Select the rating:", 1, 5, 3)

# Predict button
if st.button("Predict Sentiment"):
    if review:
        # Make prediction
        sentiment = predict_sentiment(review, hour, rating)
        # Map prediction to sentiment label
        if sentiment == 0:
            st.write(f"Predicted Sentiment: **Neutral**")
        elif sentiment == 1:
            st.write(f"Predicted Sentiment: **Positive**")
        else:
            st.write(f"Predicted Sentiment: **Negative**")
    else:
        st.warning("Please enter a review.")
