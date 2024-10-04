# Import necessary libraries
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib
import os

# Import preprocessing function from utils.py
from utils import preprocess_new

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model (ensure the model file exists in the correct location)
model = joblib.load('saved_models/best_model_xgboost.joblib')

# Route for the home page (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route for the prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Collect the form data submitted by the user
        long = float(request.form['long'])  # Longitude
        latit = float(request.form['latit'])  # Latitude
        med_age = float(request.form['med_age'])  # Median age of the house
        total_rooms = float(request.form['total_rooms'])  # Total rooms in the house
        total_bedrooms = float(request.form['total_bedrooms'])  # Total bedrooms in the house
        pop = float(request.form['pop'])  # Population in the area
        hold = float(request.form['hold'])  # Number of households
        income = float(request.form['income'])  # Median income of households
        ocean = request.form['ocean']  # Proximity to the ocean (categorical)

        # Feature engineering (based on original training data)
        rooms_per_hold = total_rooms / hold
        bedroms_per_rooms = total_bedrooms / total_rooms
        pop_per_hold = pop / hold

        # Combine input into a DataFrame (required by preprocess_new function)
        X_new = pd.DataFrame({
            'longitude': [long],
            'latitude': [latit],
            'housing_median_age': [med_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [pop],
            'households': [hold],
            'median_income': [income],
            'ocean_proximity': [ocean],
            'rooms_per_household': [rooms_per_hold],
            'bedroms_per_rooms': [bedroms_per_rooms],
            'population_per_household': [pop_per_hold]
        })

        # Preprocess the new instance (use the same pipeline from training)
        X_processed = preprocess_new(X_new)

        # Make predictions using the pre-loaded model
        y_pred_new = model.predict(X_processed)

        # Format the prediction result
        y_pred_new = '{:.4f}'.format(y_pred_new[0])

        # Render the result on the prediction page (predict.html)
        return render_template('predict.html', pred_val=y_pred_new)

    # If the request method is GET, show the prediction form
    return render_template('predict.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Run the app (in debug mode for development)
if __name__ == '__main__':
    app.run()
