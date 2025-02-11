# Movie Success Prediction Pipeline

This is a comprehensive machine learning pipeline for predicting movie success based on various features such as genre, director, stars, production companies, and more. Here's a breakdown of the key steps and components in the script:

## 1. **Data Loading and Preprocessing**
   - **Data Loading**: The script loads three datasets (`IMDb_8000.pickle`, `IMDb_top_movies.pickle`, `IMDb_top_2000s.pickle`) and concatenates them into a single DataFrame (`movies_8100`).
   - **Data Cleaning**: 
     - Drops unnecessary columns like `runtime (min)`, `opening weekend`, `writer`, etc.
     - Removes duplicate movies based on the `movie title`.
     - Drops rows with missing values in critical columns like `budget`, `imdb rating`, and `mpaa`.
     - Filters movies with at least 10,000 IMDb raters and a budget of at least $10,000.

## 2. **Feature Engineering**
   - **MPAA Ratings**: Converts the `mpaa` column into dummy variables and drops the `Not Rated` and `Unrated` categories.
   - **Genres**: 
     - Cleans the `genres` column by removing special characters and splitting genres into a list.
     - Converts the list of genres into dummy variables.
   - **Directors**:
     - Fills in missing director information.
     - Splits directors into a list and removes any aliases in parentheses.
     - Creates dummy variables for the top 10 directors.
   - **Stars**:
     - Explodes the `stars` column to handle multiple stars per movie.
     - Creates dummy variables for the top 20 stars.
   - **Production Companies**:
     - Explodes the `production companies` column to handle multiple companies per movie.
     - Creates dummy variables for the top 15 production companies.
   - **Years Since Release**: Calculates the number of years since the movie's release date.

## 3. **Target Variable Creation**
   - The target variable (`target`) is created based on the `imdb rating`:
     - `1` for ratings between 0 and 4.
     - `2` for ratings between 4 and 7.
     - `3` for ratings between 7 and 10.

## 4. **Model Training and Evaluation**
   - **Data Splitting**: The data is split into training and testing sets (80% training, 20% testing).
   - **Model Training**: Four different models are trained:
     - **Gradient Boosting Classifier**
     - **K-Nearest Neighbors (KNN)**
     - **Random Forest Classifier**
     - **Extra Trees Classifier**
   - **Model Evaluation**: The accuracy of each model is evaluated on the test set, and the results are displayed.

## 5. **Visualization**
   - **Accuracy Comparison**: A bar plot is created to compare the accuracy of the four models.
   - **Feature Importance**: A bar plot is created to show the importance of each feature according to the Random Forest model.

## 6. **Model Saving**
   - The trained Random Forest model is saved to a file (`movie_success_model.pkl`) using `pickle`.

## The endpoint `/predict`

### HTTP Method: POST

#### Request Headers:
```json
{
     "Content-Type": "application/json"
}
```

#### Request Body:
The request body should be a JSON object containing the following fields:

### `/predict` Endpoint Documentation

| Field                    | Data Type | Meaning                                                                                                                                                                                                     |
|--------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `budget`                 | `float`   | The budget of the movie (in USD or another relevant currency).                                                                                                                                              |
| `imdb raters`            | `int`     | The number of IMDb users who rated the movie.                                                                                                                                                               |
| `genres_*`               | `int`     | One-hot encoded genre columns (e.g., `genres_Action`, `genres_Comedy`, etc.). Each genre is a binary column indicating whether the movie belongs to that genre or not.                                      |
| `top_*`                  | `int`     | One-hot encoded columns for the top directors (e.g., `top_Chris_Nolan`, `top_Steven_Spielberg`). These columns represent whether the movie was directed by one of the top directors.                        |
| `production companies_*` | `int`     | One-hot encoded columns for the top production companies (e.g., `production companies_Warner Bros`, `production companies_Disney`). These columns indicate whether the movie was produced by a top company. |
| `years since release`    | `float`   | The number of years since the movie was released, calculated as the difference between the current date and the release date.                                                                               |

#### Example of One-Hot Encoded Fields:
- **Genres**: If a movie is in both Action and Comedy genres, you would have `genres_Action: 1`, `genres_Comedy: 1`, and other genre columns like `genres_Drama: 0`, `genres_Romance: 0`, etc.
- **Top Directors**: If a movie is directed by Christopher Nolan, you would have `top_Chris_Nolan: 1` and other columns like `top_Steven_Spielberg: 0`, `top_Ridley_Scott: 0`, etc.
- **Top Production Companies**: If a movie is produced by Warner Bros, you would have `production companies_Warner Bros: 1` and other columns like `production companies_Disney: 0`, `production companies_Universal: 0`, etc.

#### Example Request Body

```json
{
  "budget": [50000000],
  "imdb raters": [200000],
  "genres_Action": [1],
  "genres_Comedy": [0],
  "genres_Drama": [0],
  "genres_Romance": [1],
  "top_Chris_Nolan": [1],
  "top_Steven_Spielberg": [0],
  "production companies_Warner Bros": [1],
  "production companies_Disney": [0],
  "years since release": [5]
}
```

## Key Points:
- **Data Cleaning**: The script handles missing data, duplicates, and irrelevant columns effectively.
- **Feature Engineering**: The script creates a rich set of features by encoding categorical variables and creating interaction terms.
- **Model Selection**: Multiple models are trained and evaluated, allowing for a comparison of their performance.
- **Visualization**: The script includes visualizations to help interpret the results and understand feature importance.
- **Model Persistence**: The best-performing model is saved for future use.

## Potential Improvements:
- **Hyperparameter Tuning**: The models could be further improved by tuning their hyperparameters using techniques like Grid Search or Random Search.
- **Cross-Validation**: Instead of a single train-test split, cross-validation could be used to get a more robust estimate of model performance.
- **Feature Selection**: Further feature selection could be performed to reduce dimensionality and potentially improve model performance.
- **Handling Imbalanced Data**: If the target classes are imbalanced, techniques like SMOTE or class weighting could be applied.
