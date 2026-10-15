import pandas as pd
import joblib

print("1. Loading the saved AI model brain...")
# Load the saved rules back into memory
model = joblib.load('heart_model.pkl')

# ----------------------------------------------------
# You can change these numbers to test different people:
input_age = 45
input_exercise = 7
# ----------------------------------------------------

print(f"2. Predicting heart rate for Age: {input_age}, Exercise: {input_exercise} hours...")

# Convert the input numbers into a structured table format
new_data = pd.DataFrame({'Age': [input_age], 'Exercise_Hours': [input_exercise]})

# Ask the AI model to guess the target value
guessed_hr = model.predict(new_data)

print(f"\n🔮 AI Prediction Result ➡️ Heart Rate: {guessed_hr[0]:.1f} bpm")
