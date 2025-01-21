from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('movie_success_model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Define columns used during training
TRAINING_COLUMNS = [
    'budget', 'imdb raters', 'genres_Action', 'genres_Comedy', 'genres_Drama', 'genres_Romance', 
    'top_Chris_Nolan', 'top_Steven_Spielberg', 'production companies_Warner Bros', 'years since release'
]

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
    data = request.get_json()

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame(data)

    # Align input data columns to match the training columns
    input_data = input_data.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    # Make prediction using the trained model
    prediction = model.predict(input_data)

    print('input_data:', input_data)
    print('prediction:', prediction)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
