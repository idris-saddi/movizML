import pandas as pd
from utils import directors_list, remove_paren, extract_top_n, encode_top_n
import numpy as np

categorized_features = dict()


# Load datasets
def load_data():
    """
    Loads the movie datasets from pickle files and ensures they are DataFrames.
    """
    movies_8000 = pd.read_pickle('./data/IMDb_8000.pickle')
    df1 = pd.read_pickle('./data/IMDb_top_movies.pickle')
    df2 = pd.read_pickle('./data/IMDb_top_2000s.pickle')

    # Ensure each variable is a DataFrame
    movies_8000 = pd.DataFrame(movies_8000)
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)

    # Concatenate the DataFrames
    movies = pd.concat([movies_8000, df1, df2], ignore_index=True)
    return movies


# Clean and preprocess the dataset
def preprocess_data(movies):
    """
    Cleans the dataset by removing unnecessary columns, handling missing values,
    and filtering movies based on specific criteria.
    """
    # Drop unnecessary columns
    movies = movies.drop(['runtime (min)', 'opening weekend', 'writer', 'country', 'language', 'gross usa'], axis=1)

    # Remove duplicates based on movie title
    movies.drop_duplicates(subset=['movie title'], inplace=True)
    movies.set_index('movie title', inplace=True)

    # Drop rows with missing budget
    movies = movies.dropna(subset=['budget'])

    # Keep movies with MPAA ratings and sufficient ratings/budget
    movies = movies[movies['mpaa'].notna() & (movies['imdb raters'] >= 10000) & (movies['budget'] >= 10000)]

    return movies


# Handle categorical features (MPAA ratings, genres, etc.)
def encode_features(movies):
    """
    Encodes categorical features such as MPAA ratings and genres into numerical format.
    """
    uncategorized_feats = ['release date', 'genres', 'imdb raters', 'director', 'stars', 'production companies', 'mpaa']
    numerical_feats = [col for col in movies.columns if col not in uncategorized_feats]
    numerical_feats.append('years since release')

    # Convert MPAA ratings to dummy variables
    mpaa_dummies = pd.get_dummies(movies['mpaa']).drop(columns=['Not Rated', 'Unrated'], errors='ignore')
    movies = movies.drop(columns=['mpaa']).join(mpaa_dummies)

    # Process genres
    movies['genres'] = movies['genres'].fillna('').astype(str).str.replace(r"[\"\[\]\']", "", regex=True)

    # Split genres by comma, strip spaces from each genre, and explode
    genre_list = movies['genres'].str.split(',').apply(lambda x: [i.strip() for i in x])
    genre_dummies = pd.get_dummies(genre_list.explode()).groupby(level=0).sum()

    # Merge genres into the main DataFrame
    movies = movies.drop(columns=['genres']).join(genre_dummies)

    # Drop 'imdb raters' column if present
    movies = movies.drop(columns=['imdb raters'], errors='ignore')

    categorized_features.update({'genres': list(genre_dummies.columns), 'ratings': list(mpaa_dummies.columns),
                                 'numerical': numerical_feats})

    return movies


# Compute years since release
def compute_years_since_release(movies):
    """
    Computes the number of years since a movie's release date.
    """
    current_date = pd.Timestamp.today()  # Use Pandas Timestamp
    movies['release date'] = pd.to_datetime(movies['release date'], errors='coerce')  # Convert safely
    movies['years since release'] = (current_date - movies['release date']).dt.days / 365.25  # Convert to years
    # Drop 'release date' to avoid datetime issues
    movies = movies.drop(columns=['release date'], errors='ignore')

    return movies


# Process director names
def process_directors(movies):
    """
    Cleans and processes the director column by splitting names and removing aliases.
    """
    movies['director'] = movies['director'].apply(directors_list).apply(remove_paren)
    return movies


# Process top N directors, stars, and production companies
def process_top_entities(movies):
    """
    Identifies the top directors, stars, and production companies and encodes them as dummy variables.
    """
    top_directors = extract_top_n(movies, 'director', 10)
    movies, director_names = encode_top_n(movies, 'director', top_directors)

    top_stars = extract_top_n(movies, 'stars', 20)
    movies, star_names = encode_top_n(movies, 'stars', top_stars)

    top_production_companies = extract_top_n(movies, 'production companies', 15)
    movies, production_names = encode_top_n(movies, 'production companies', top_production_companies)

    categorized_features.update(director_names | star_names | production_names)

    return movies


# Main function to prepare data
def prepare_data():
    """
    Loads, processes, and encodes the movie dataset, then returns the cleaned DataFrame.
    """
    movies = load_data()
    movies = preprocess_data(movies)
    movies = encode_features(movies)
    movies = compute_years_since_release(movies)
    movies = process_directors(movies)
    movies = process_top_entities(movies)

    # Ensure 'imdb rating' exists before creating the target column
    if 'imdb rating' not in movies.columns:
        raise ValueError("Error: 'imdb rating' column is missing from the dataset.")

    # Define classification target based on IMDb rating
    criteria = [
        movies['imdb rating'].between(0, 4),  # Low ratings
        movies['imdb rating'].between(4, 7),  # Medium ratings
        movies['imdb rating'].between(7, 10)  # High ratings
    ]
    values = [1, 2, 3]  # 1: Low, 2: Medium, 3: High ratings
    movies['target'] = np.select(criteria, values, default=0)

    # Drop IMDb rating AFTER defining the target variable
    movies = movies.drop(columns=['imdb rating'], errors='ignore')
    movies.columns = movies.columns.str.strip()

    return movies, categorized_features


if __name__ == "__main__":
    processed_data, feature_names = prepare_data()
    processed_data.to_pickle("./model/processed_movies.pkl")
