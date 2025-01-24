
import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

log_reg_model = joblib.load('log_reg_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(review, hour, rating):
    review_cleaned = review.lower()
    review_tfidf = tfidf_vectorizer.transform([review_cleaned])
    numeric_features = np.array([[hour, rating]])
    features = hstack([review_tfidf, numeric_features])
    prediction = log_reg_model.predict(features)
    return prediction[0]

st.title("TikTok Review Sentiment Analysis")

review = st.text_input("Enter your review:")
hour = st.slider("Select the hour of the day:", 0, 24, 12)
rating = st.slider("Select the rating:", 1, 5, 3)

if st.button("Predict Sentiment"):
    if review:
        sentiment = predict_sentiment(review, hour, rating)
        if sentiment == 0:
            st.write(f"Predicted Sentiment: **Neutral**")
        elif sentiment == 1:
            st.write(f"Predicted Sentiment: **Positive**")
        else:
            st.write(f"Predicted Sentiment: **Negative**") 
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("Please enter a review.")
