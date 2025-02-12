import requests
import json


def test_get_features():
    res = requests.get("http://127.0.0.1:5000/features")
    print('/features  GET')
    print(f"Status code: {res.status_code}")
    print(f"Response: {res.json()}")

    # Flattening the JSON by combining all the list elements
    feature_names = res.json()['features']
    flattened_feature_names = [item for sublist in feature_names.values() for item in sublist]
    return flattened_feature_names


def test_predict(flattened_feature_names):
    # URL of the Flask API
    url = "http://127.0.0.1:5000/predict"

    # Create a dictionary with all expected features, defaulting to 0
    data = {feature: 0 for feature in flattened_feature_names}

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
    print('/predict POST')
    print("Status Code:", response.status_code)
    print("Response:", response.json())


if __name__ == '__main__':
    features = test_get_features()
    test_predict(features)
