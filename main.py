from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('./model/refactored_success_model.pkl', 'rb'))
feature_names = pickle.load(open('./model/feature_names.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)


# Function to add CORS headers manually
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:4200'  # Replace with your Angular app URL
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


# Handle OPTIONS method for preflight requests
@app.route('/predict', methods=['OPTIONS'])
def options():
    return jsonify({}), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Create a dictionary with all expected features, defaulting to 0
        input_dict = {feature: 0 for feature in feature_names}
        # Update the dictionary with the provided data
        input_dict.update(data)
        # Convert the input data into a DataFrame
        input_data = pd.DataFrame([input_dict], columns=feature_names)
        # Make prediction using the trained model
        prediction = model.predict(input_data)
        print('prediction:', prediction)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        print(f"Prediction error : {e}")
        return jsonify({"error": "An error occured during preduction"}), 500


if __name__ == '__main__':
    app.run(debug=True)
