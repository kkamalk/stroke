import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib

# Load the pre-trained model
model = joblib.load('best_model.pkl')

# Create the Flask app
app = Flask(__name__)

# Root route
@app.route('/')
def index():
    return 'Welcome to the Flask application!'

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()

    # Convert input data to DataFrame
    df = pd.DataFrame(data)

    # Perform any necessary validation or data preprocessing on the input data
    # ...

    # Convert DataFrame to numpy array
    X = df.values

    # Make predictions using the loaded model
    predictions = model.predict(X)

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    # Run the Flask app on localhost:5000
    app.run(host='localhost', port=5000, debug=True)
