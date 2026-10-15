# Heart Rate Prediction using Machine Learning

A beginner-friendly Machine Learning project built in Python using Scikit-Learn.

## Project Overview
This project uses a **Linear Regression** algorithm to predict an individual's resting heart rate (bpm) based on two basic features:
1. **Age** (Years)
2. **Weekly Exercise** (Hours)

*Note: This project utilizes simulated data designed to demonstrate a clean, end-to-end Machine Learning pipeline.*

## Project Structure
* `dataset.csv` - The training dataset containing profile examples.
* `train.py` - Reads the dataset, trains the Linear Regression model, and exports the model state.
* `heart_model.pkl` - The trained AI model brain file.
* `predict.py` - Loads the trained model and processes user custom inputs to deliver a heart rate prediction.

## How to Run
1. Install requirements: `pip install pandas scikit-learn joblib`
2. Train the model: `python train.py`
3. Make predictions: `python predict.py`
# Heart-Rate-Prediction-ML
