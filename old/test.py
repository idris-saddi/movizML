import requests
import json

# URL of the Flask API
url = 'http://127.0.0.1:5000/predict'

# Sample input data (replace with actual data that matches your model's input features)
data = {
    "cumulative worldwide": 0,
    "G": 0,
    "PG": 1,
    "PG-13": 0,
    "R": 0,
    "TV-MA": 0,
    "Action": 1,
    "Adventure": 1,
    "Animation": 0,
    "Biography": 0,
    "Comedy": 0,
    "Crime": 0,
    "Drama": 0,
    "Family": 0,
    "Fantasy": 0,
    "History": 0,
    "Horror": 0,
    "Music": 0,
    "Musical": 0,
    "Mystery": 0,
    "Romance": 0,
    "Sci-Fi": 0,
    "Sport": 0,
    "Thriller": 0,
    "War": 0,
    "Western": 0,
    "Brad Bird": 0,
    "Christopher Nolan": 1,
    "Clint Eastwood": 0,
    "Denis Villeneuve": 0,
    "Deon Taylor": 0,
    "Martin Scorsese": 0,
    "Paul Feig": 0,
    "Quentin Tarantino": 0,
    "Ron Howard": 0,
    "Steven Spielberg": 0,
    "Dwayne Johnson": 0,
    "Matthew McConaughey": 0,
    "Kevin Hart": 0,
    "Tom Hanks": 0,
    "Margot Robbie": 0,
    "Samuel L. Jackson": 0,
    "Jake Gyllenhaal": 0,
    "Leonardo DiCaprio": 1,
    "Anna Kendrick": 0,
    "Will Smith": 0,
    "Michael Fassbender": 0,
    "Mark Wahlberg": 0,
    "Ryan Reynolds": 0,
    "Joel Edgerton": 0,
    "Matt Damon": 0,
    "Charlize Theron": 0,
    "Jessica Chastain": 0,
    "Steve Carell": 0,
    "Nicole Kidman": 0,
    "Woody Harrelson": 0,
    "Columbia Pictures": 0,
    "Universal Pictures": 1,
    "Warner Bros.": 0,
    "Walt Disney Pictures": 0,
    "Paramount Pictures": 0,
    "New Line Cinema": 0,
    "Twentieth Century Fox": 0,
    "Blumhouse Productions": 0,
    "Summit Entertainment": 0,
    "Perfect World Pictures": 0,
    "TSG Entertainment": 0,
    "Legendary Entertainment": 0,
    "Metro-Goldwyn-Mayer (MGM)": 0,
    "Lionsgate": 0,
    "LStar Capital": 0,
    "years since release": 5
}

# Convert the data to JSON format
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

# Print the response from the server
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
