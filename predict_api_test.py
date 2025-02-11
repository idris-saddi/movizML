import requests
import json
import pickle

# Load the exact feature names used in training
with open("./model/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# URL of the Flask API
url = "http://127.0.0.1:5000/predict"

# Create a dictionary with all expected features, defaulting to 0
data = {feature: 0 for feature in feature_names}

# Modify some features for testing
data.update({
    "cumulative worldwide": 100000000,  # Example revenue
    "PG-13": 1,  # Assume it's a PG-13 movie
    "Action": 1,  # Action genre
    "Adventure": 1,
    "Christopher Nolan": 1,  # Directed by Nolan
    "Leonardo DiCaprio": 1,  # Actor present
    "Warner Bros.": 1,  # Production company
    "years since release": 5  # Released 5 years ago
})

# Convert to JSON format
headers = {"Content-Type": "application/json"}
response = requests.post(url, data=json.dumps(data), headers=headers)

# Print response
print("Status Code:", response.status_code)
print("Response:", response.json())
