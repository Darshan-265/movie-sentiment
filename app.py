import streamlit as st
import joblib
import re

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("🎬 IMDb Movie Review Sentiment Analyzer")
st.write("Enter your movie review below:")

user_input = st.text_area("✍️ Write your review here")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"🧠 Sentiment: **{prediction.upper()}**")
