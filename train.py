import json
from data_processing import prepare_data
from model_training import load_processed_data, prepare_training_data, plot_feature_importance
from model_training import plot_model_comparison, train_and_evaluate_models, save_model

if __name__ == "__main__":
    # ///////////// Process and save the cleaned dataset /////////////
    print("Preparing data...")
    processed_data, categorized_features = prepare_data()
    with open("model/categorized_features.json", "w") as file:
        json.dump(categorized_features, file, indent=4)
    processed_data.to_pickle("./model/processed_movies.pkl")
    # ///////////// ///////////// ///////////// ///////////// /////////////

    print("Loading processed data...")
    df = load_processed_data()

    print("Splitting dataset...")
    x_train, x_test, y_train, y_test = prepare_training_data(df)

    print("Training models...")
    best_model, results = train_and_evaluate_models(x_train, x_test, y_train, y_test)

    print("Plotting model performance...")
    plot_model_comparison(results)

    # After model training
    print("Plotting feature importance...")
    plot_feature_importance(best_model, x_train.columns)

    print("Saving best model...")
    save_model(best_model, "./model/refactored_success_model.pkl")

    print("Training complete. Model saved successfully!")
