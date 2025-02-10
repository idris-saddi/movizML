from datetime import datetime
import pandas as pd


# Load datasets
def load_data():
    """
    Loads the movie datasets from pickle files and combines them into a single DataFrame.
    """
    movies_8000 = pd.read_pickle('./data/IMDb_8000.pickle')
    df1 = pd.read_pickle('./data/IMDb_top_movies.pickle')
    df2 = pd.read_pickle('./data/IMDb_top_2000s.pickle')
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
    # Convert MPAA ratings to dummy variables
    mpaa_dummies = pd.get_dummies(movies['mpaa']).drop(columns=['Not Rated', 'Unrated'], errors='ignore')
    movies = movies.drop(columns=['mpaa']).join(mpaa_dummies)

    # Process genres
    movies['genres'] = movies['genres'].fillna('').astype(str).str.replace(r"[\"\[\]\']", "", regex=True).str.strip()
    genre_dummies = pd.get_dummies(movies['genres'].str.split(',').apply(pd.Series).stack()).groupby(level=0).sum()

    # Merge genres into the main DataFrame
    movies = movies.drop(columns=['genres']).join(genre_dummies)

    # Drop 'imdb raters' column if present
    movies = movies.drop(columns=['imdb raters'], errors='ignore')

    return movies


# Compute years since release
def compute_years_since_release(movies):
    """
    Computes the number of years since a movie's release date.
    """
    current_date = pd.to_datetime(datetime.now().date())
    movies['years since release'] = movies['release date'].apply(
        lambda x: (current_date - pd.to_datetime(x)).days / 365.25)
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
    return movies


if __name__ == "__main__":
    processed_data = prepare_data()
    processed_data.to_pickle("./data/processed_movies.pkl")
