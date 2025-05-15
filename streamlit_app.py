import streamlit as st
import joblib

# Load model
model = joblib.load("emotion_model.joblib")

st.title("Emotion Detection from Text")
st.markdown("Type a sentence and I’ll predict the emotion!")

# Input
user_input = st.text_input("Enter your sentence:")

if st.button("Detect Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction = model.predict([user_input])[0]
        st.success(f"Predicted Emotion: **{prediction.capitalize()}**")
