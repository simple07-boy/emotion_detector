import joblib

# Load the model
model = joblib.load("emotion_model.joblib")

# User input
sentence = input("Enter a sentence: ")
emotion = model.predict([sentence])[0]

print(f"\nDetected Emotion: {emotion}")