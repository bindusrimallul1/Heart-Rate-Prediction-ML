import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

print("1. Reading the dataset file (dataset.csv)...")
# Load the data table
df = pd.read_csv('dataset.csv')

# Split data into Clues (X) and Answers (y)
X = df[['Age', 'Exercise_Hours']]
y = df['Heart_Rate']

print("2. Training the AI model to learn the rules...")
# Initialize the algorithm and train it using the data
model = LinearRegression()
model.fit(X, y)

print("3. Saving the learned model rules to 'heart_model.pkl'...")
# Save the model brain to a file for future use
joblib.dump(model, 'heart_model.pkl')

print("🎉 Success! The model training is complete and saved.")
