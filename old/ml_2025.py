# import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# warnings.filterwarnings('ignore')

movies_8000 = pd.DataFrame(pd.read_pickle('../data/IMDb_8000.pickle'))
df1 = pd.DataFrame(pd.read_pickle('../data/IMDb_top_movies.pickle'))
df2 = pd.DataFrame(pd.read_pickle('../data/IMDb_top_2000s.pickle'))
movies_8100 = pd.concat([movies_8000, df1, df2])
movies_8100.head()

movies_8100 = movies_8100.drop(
    ['runtime (min)', 'opening weekend', 'writer', 'country', 'language', 'opening weekend', 'gross usa'], axis=1)

movies_df = movies_8100.copy()
movies_df.drop_duplicates(subset=['movie title'], inplace=True)
movies_df.set_index('movie title', inplace=True)

movies_df_drop = movies_df.dropna(subset=['budget'])
# movie_drop2 = movies_df_drop[movies_df_drop['imdb rating'].notna()]
movie_drop2 = movies_df_drop[movies_df_drop['mpaa'].notna()]

movie_drop2 = movie_drop2[movie_drop2['imdb raters'] >= 10000]
movie_drop2 = movie_drop2[movie_drop2['budget'] >= 10000]

mpaa_df = pd.get_dummies(movie_drop2['mpaa'])
df_added_mpaa_dummies = pd.concat([movie_drop2, mpaa_df], axis=1)
df_added_mpaa_dummies = df_added_mpaa_dummies.drop('mpaa', axis=1)
df_added_mpaa_dummies = df_added_mpaa_dummies.drop(['Not Rated', 'Unrated'], axis=1)
df_added_mpaa_dummies.head()

df_added_mpaa_dummies['genres'] = df_added_mpaa_dummies['genres'].fillna('').astype(str)

df_added_mpaa_dummies['genres'] = (
    df_added_mpaa_dummies['genres']
    .str.replace(r"[\"\[\]\']", "", regex=True)
    .str.strip()
)

df_genres_added = df_added_mpaa_dummies['genres'].str.split(',').apply(
    lambda x: [genre.strip() for genre in x if genre.strip()])
df_genres_dummies = pd.get_dummies(df_genres_added.apply(pd.Series).stack()).groupby(level=0).sum()

df_genres_mpaa = pd.concat([df_added_mpaa_dummies.drop(columns=['genres']), df_genres_dummies], axis=1)

if 'imdb raters' in df_genres_mpaa.columns:
    df_genres_mpaa = df_genres_mpaa.drop(columns=['imdb raters'])

df_genres_mpaa.head()

empty_dir = movie_drop2[movie_drop2['director'] == ''].index.values.tolist()
dir_fill = [
    'Peter Ramsey, Bob Persichetti, Rodney Rothman',
    'Ron Clements, John Musker',
    'Byron Howard, Rich Moore',
    'Pierre Coffin, Chris Renaud',
    'Charlie Bean, Bob Logan, Paul Fisher',
    'Frank Miller, Quentin Tarantino, Robert Rodriguez',
    'Dan Scanlon, Saschka Unseld'
]
for num, movie in enumerate(empty_dir):
    movie_drop2.loc[movie, 'director'] = dir_fill[num]


# Run through the empty directors and fill with added info


def directors_list(directors):
    """
    Separates a string of directors into a list of the separate directors.
    Args:
        directors: A string of directors separated by commas
    Returns:
        A list of directors.
    """
    if "," in directors:
        return [name.strip() for name in directors.split(",")]
    else:
        return [directors]


def remove_paren(directors):
    """
    Takes off aliases that are in parentheses next to a director's name
    Args:
        directors: A list of directors
    Returns:
        The same list of directors, but without aliases in parentheses.
    """
    dir_list = []
    for director in directors:
        if "(" in director:
            dir_clean = director.split("(")[0].strip()
            dir_list.append(dir_clean)
        else:
            dir_list.append(director)
    return dir_list


movie_drop2['director'] = movie_drop2['director'].apply(lambda x: directors_list(x))
movie_drop2['director'] = movie_drop2['director'].apply(lambda x: remove_paren(x))

df_dir_exploded = movie_drop2.explode('director')
dir_10 = df_dir_exploded['director'].value_counts()[:10].index.tolist()


def top_directors(directors):
    dir_list = []
    for director in directors:
        if director in dir_10:
            dir_list.append(director)
    return dir_list


# Create a function to select top directors and create new columns
# This will let us create dummy variables just for top directors

movie_drop2['top'] = movie_drop2['director'].apply(lambda x: top_directors(x))
# Apply that function to create a new column
dir_df = pd.get_dummies(movie_drop2['top'].apply(pd.Series).stack()).groupby(level=0).sum()
# Get dummies for the top directors
df_dir_model = pd.concat([df_genres_mpaa, dir_df], axis=1)
# Merge dummies with model DataFrame
df_dir_model.replace(np.nan, 0, inplace=True)

stars_exploded = movie_drop2.explode('stars')
# Let's do something similar to director
stars_20 = stars_exploded['stars'].value_counts()[:20].index.tolist()


# Let's take the top 20 stars instead of just 10
def top_stars(stars):
    star_list = []
    for star in stars:
        if star in stars_20:
            star_list.append(star)
    return star_list


# Create a function to select top stars and create new columns
# This will let us create dummy variables just for top stars
movie_drop2['top stars'] = movie_drop2['stars'].apply(lambda x: top_stars(x))
stars_df = pd.get_dummies(movie_drop2['top stars'].apply(pd.Series).stack()).groupby(level=0).sum()
star_model = pd.concat([df_dir_model, stars_df[stars_20]], axis=1)
star_model.replace(np.nan, 0, inplace=True)

prod_exploded = movie_drop2.explode('production companies')
# You know the drill
prod_15 = prod_exploded['production companies'].value_counts()[:15].index.tolist()
movie_drop2.loc[movie_drop2['production companies'].isnull(), 'production companies'] = movie_drop2.loc[
    movie_drop2['production companies'].isnull(), 'production companies'].apply(lambda x: [])


def top_prod(comp):
    comp_list = []
    for co in comp:
        if co in prod_15:
            comp_list.append(co)
    return comp_list


movie_drop2['top prod co.'] = movie_drop2['production companies'].apply(lambda x: top_prod(x))
prod_df = pd.get_dummies(movie_drop2['top prod co.'].apply(pd.Series).stack()).groupby(level=0).sum()
prod_model = pd.concat([star_model, prod_df[prod_15]], axis=1)
prod_model.replace(np.nan, 0, inplace=True)

date = pd.to_datetime(datetime.now().date())
movie_drop2['years since release'] = movie_drop2['release date'].apply(
    lambda x: ((date - pd.to_datetime(x)).days / 365.25))
prod_model['years since release'] = movie_drop2['years since release']
prod_model = prod_model.drop(['director', 'stars', 'release date', 'production companies'], axis=1)

criteria = [prod_model['imdb rating'].between(0, 4), prod_model['imdb rating'].between(4, 7),
            prod_model['imdb rating'].between(7, 10)]
values = [1, 2, 3]
prod_model['target'] = np.select(criteria, values, 0)
prod_model.tail()
prod_model = prod_model.drop('imdb rating', axis=1)

X, y = prod_model.iloc[:, 1:73], prod_model['target']
cols = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=20)

rf = RandomForestClassifier(n_estimators=1000)
KNN = KNeighborsClassifier()
Extra = ExtraTreesClassifier()
GD = GradientBoostingClassifier()

GD.fit(X_train, y_train)
gd = GD.score(X_test, y_test)
print(f'GradientBoostingClassifier: :', gd * 100)

rf.fit(X_train, y_train)
RF = rf.score(X_test, y_test)
print(f'Random forest:', RF * 100)

KNN.fit(X_train, y_train)
knn = KNN.score(X_test, y_test)
print(f'KNN: ', knn * 100)

Extra.fit(X_train, y_train)
extra = Extra.score(X_test, y_test)
print(f'Extratrees: ', extra * 100)

accuracy = ['Gradient Boosting', 'KNN', 'Random Forest', 'Extra Trees']
values = [75.3, 65.1, 82.5, 77.1]
plt.figure(figsize=(10, 6))
plt.bar(accuracy, values)
# Displaying the bar plot
plt.xlabel('Classification algorithm')
plt.ylabel('Accuracy%')
plt.title('Comparison between different algorithms accuracy')
plt.show()

# rf is our trained RandomForestClassifier
importances = rf.feature_importances_
feature_names = X.columns  # Assuming 'X' is your feature matrix

eval_df1 = pd.DataFrame({'Feature': feature_names, 'Importances': importances})

# Sort the DataFrame by importance scores in descending order
eval_df1 = eval_df1.sort_values(by=['Importances'], ascending=False)

ax1 = eval_df1.plot(x='Feature', y='Importances', kind='bar', figsize=(20, 10), fontsize=13)
plt.title('Feature Importances', fontsize=15)
ax1.set_xlabel('Features', fontsize=13)
ax1.set_ylabel('Importance Scores by sklearn', fontsize=13)
plt.show()

with open('movie_success_model.pkl', 'wb') as file:
    pickle.dump(rf, file)

print("Features required by the model:", list(X.columns))
