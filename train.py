from data_processing import prepare_data
from model_training import load_processed_data, prepare_training_data, plot_feature_importance
from model_training import plot_model_comparison, train_and_evaluate_models, save_model

if __name__ == "__main__":
    print("Preparing data...")
    # Process and save the cleaned dataset
    processed_data = prepare_data()
    processed_data.to_pickle("./data/processed_movies.pkl")

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
    save_model(best_model)

    print("Training complete. Model saved successfully!")
