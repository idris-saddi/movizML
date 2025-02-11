from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Load processed data
def load_processed_data():
    """Loads the processed movie dataset from pickle file."""
    return pd.read_pickle("./model/processed_movies.pkl")


# Prepare data for model training
def prepare_training_data(dataf):
    """
    Splits the dataset into features (X) and target variable (y), then creates train-test splits.
    """
    x = dataf.drop(columns=['target'])
    y = dataf['target']
    # ---Fill missing values with 0
    x = x.fillna(0)
    # ---Alternative:
    # x = x.fillna(x.mean())
    # ---Alternatively drop rows with missing values
    # x = x.dropna()
    # y = y.loc[x.index]  # Keep only the corresponding y-values

    return train_test_split(x, y, test_size=0.2, random_state=20)


# Train and evaluate models
def train_and_evaluate_models(x_train, x_test, y_train, y_test):
    """
    Trains multiple classifiers and evaluates them on the test dataset.
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=1000),
        'KNN': KNeighborsClassifier(),
        'Extra Trees': ExtraTreesClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    res = {}
    optimal_model = None
    best_score = 0
    for name, model in models.items():
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test) * 100
        res[name] = score
        print(f'{name} Accuracy: {score:.2f}%')
        if score > best_score:
            best_score = score
            optimal_model = model

    return optimal_model, res


# Plot model comparison
def plot_model_comparison(res):
    """Plots a bar chart comparing model accuracies."""
    plt.figure(figsize=(10, 6))
    plt.bar(res.keys(), res.values(), color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('Classification Algorithm')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparison of Model Accuracies')
    plt.show()


# Save the best model
def save_model(model, filename="movie_success_model.pkl"):
    """Saves the trained model to a file using pickle."""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}")


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plots feature importance for tree-based models, showing only the top N most important features.
    """
    if hasattr(model, "feature_importances_"):  # Ensure model supports feature importance
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Limit to top N features
        feature_importance_df = feature_importance_df.head(top_n)

        plt.figure(figsize=(12, 8))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()  # Most important features at the top

        # Adjust layout to prevent overlap
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        with open('./model/feature_names.pkl', 'wb') as f:
            pickle.dump(list(feature_names), f)
    else:
        print("Feature importance not available for this model.")


if __name__ == "__main__":
    df = load_processed_data()
    X_train, X_test, Y_train, Y_test = prepare_training_data(df)
    best_model, results = train_and_evaluate_models(X_train, X_test, Y_train, Y_test)
    plot_model_comparison(results)
    save_model(best_model)
    plot_feature_importance(best_model, X_train.columns)
