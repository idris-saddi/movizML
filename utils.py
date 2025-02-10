import pandas as pd


# Utility functions for cleaning and processing director names
def directors_list(directors):
    """
    Converts a string of directors into a list of individual names.
    """
    return [name.strip() for name in directors.split(",")] if "," in directors else [directors]


def remove_paren(directors):
    """
    Removes aliases in parentheses from director names.
    """
    return [director.split("(")[0].strip() for director in directors]


# Function to extract top N frequent values from a column
def extract_top_n(df, column, n):
    """
    Extracts the top N most frequent values from a specified column.
    """
    return df[column].explode().value_counts().nlargest(n).index.tolist()


# Function to create dummy variables for top N values in a column
def encode_top_n(df, column, top_n_values):
    """
    Encodes the presence of top N values as dummy variables in a DataFrame.
    """
    df[column] = df[column].apply(lambda x: x if isinstance(x, list) else [])  # Ensure all values are lists
    df[f'top {column}'] = df[column].apply(lambda x: [item for item in x if item in top_n_values])
    dummies = pd.get_dummies(df[f'top {column}'].apply(pd.Series).stack()).groupby(level=0).sum()
    return df.drop(columns=[column, f'top {column}']).join(dummies, how='left')


if __name__ == "__main__":
    print("Utility functions for movie data processing.")
