# Imports Explanation
#Imports Flask for API server (Flask = web framework; request = handles input; jsonify = formats JSON output), Joblib for loading the saved model (Joblib = serialization tool for ML models).

from flask import Flask, request, jsonify  # Flask creates the app; request gets input data; jsonify formats output as JSON.
import joblib  # Joblib loads the saved model from Project 1 (efficient for scikit-learn models).
import pandas as pd

# Model Loading and API Explanation
#Loads the saved Iris model; defines /predict endpoint to accept feature input and return species prediction.

app = Flask(__name__)  # Initializes the Flask application; __name__ is the module name for routing.
model = joblib.load('iris_model.pkl')  # Loads the saved Iris model from Project 1; joblib.load deserializes it.

@app.route('/predict', methods=['POST'])  # Decorator defines the /predict URL endpoint; methods=['POST'] allows JSON input.
def predict():  # Prediction function.
    data = request.get_json()  # Gets JSON input.
    features_df = pd.DataFrame([[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]], 
                               columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])  # DF with names.
    prediction = int(model.predict(features_df)[0])  # Predict and convert to int.
    return jsonify({'prediction': prediction})  # Returns JSON.
if __name__ == '__main__':  # Checks if script is main (not imported).
    app.run(debug=True, port=5000)  # Runs the Flask server in debug mode (port 5000 local); debug=True shows errors.

