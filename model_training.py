from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# Load processed data
def load_processed_data():
    """Loads the processed movie dataset from pickle file."""
    return pd.read_pickle("./data/processed_movies.pkl")


# Prepare data for model training
def prepare_training_data(dataf):
    """
    Splits the dataset into features (X) and target variable (y), then creates train-test splits.
    """
    x = dataf.drop(columns=['target'])
    y = dataf['target']
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
    for name, model in models.items():
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test) * 100
        res[name] = score
        print(f'{name} Accuracy: {score:.2f}%')

    return models['Random Forest'], res


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


if __name__ == "__main__":
    df = load_processed_data()
    X_train, X_test, Y_train, Y_test = prepare_training_data(df)
    best_model, results = train_and_evaluate_models(X_train, X_test, Y_train, Y_test)
    plot_model_comparison(results)
    save_model(best_model)
