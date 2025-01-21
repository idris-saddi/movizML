from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('movie_success_model.pkl', 'rb'))

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
    data = request.get_json()

    # Define the expected features based on the model
    expected_features = [
        'cumulative worldwide', 'G', 'PG', 'PG-13', 'R', 'TV-MA', 'Action', 'Adventure', 'Animation', 'Biography',
        'Comedy', 'Crime', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western', 'Brad Bird', 'Christopher Nolan', 'Clint Eastwood',
        'Denis Villeneuve', 'Deon Taylor', 'Martin Scorsese', 'Paul Feig', 'Quentin Tarantino', 'Ron Howard',
        'Steven Spielberg', 'Dwayne Johnson', 'Matthew McConaughey', 'Kevin Hart', 'Tom Hanks', 'Margot Robbie',
        'Samuel L. Jackson', 'Jake Gyllenhaal', 'Leonardo DiCaprio', 'Anna Kendrick', 'Will Smith', 'Michael Fassbender',
        'Mark Wahlberg', 'Ryan Reynolds', 'Joel Edgerton', 'Matt Damon', 'Charlize Theron', 'Jessica Chastain',
        'Steve Carell', 'Nicole Kidman', 'Woody Harrelson', 'Columbia Pictures', 'Universal Pictures', 'Warner Bros.',
        'Walt Disney Pictures', 'Paramount Pictures', 'New Line Cinema', 'Twentieth Century Fox', 'Blumhouse Productions',
        'Summit Entertainment', 'Perfect World Pictures', 'TSG Entertainment', 'Legendary Entertainment',
        'Metro-Goldwyn-Mayer (MGM)', 'Lionsgate', 'LStar Capital', 'years since release'
    ]

    # Create a dictionary with all expected features, defaulting to 0
    input_dict = {feature: 0 for feature in expected_features}

    # Update the dictionary with the provided data
    input_dict.update(data)

    # Convert the input data into a DataFrame
    input_data = pd.DataFrame([input_dict])

    # Make prediction using the trained model
    prediction = model.predict(input_data)

    print('prediction:', prediction)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)