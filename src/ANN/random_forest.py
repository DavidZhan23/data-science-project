import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_preprocessing import load_and_preprocess_data

def train_and_evaluate_rf(X_train, X_test, y_train, y_test, best_params):
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

if __name__ == "__main__":
    file_path = '../../cleaned_dataset/data4training.csv'

    with open("best_params.json", "r") as f:
        best_params = json.load(f)

    X_train_transformed, X_test_transformed, y_train_novelty, y_test_novelty, y_train_usefulness, y_test_usefulness = load_and_preprocess_data(file_path)

    print("Evaluating Random Forest for novelty...")
    novelty_report = train_and_evaluate_rf(X_train_transformed, X_test_transformed, y_train_novelty, y_test_novelty, best_params['novelty'])
    print("Novelty Prediction Report")
    print(novelty_report)

    print("Evaluating Random Forest for usefulness...")
    usefulness_report = train_and_evaluate_rf(X_train_transformed, X_test_transformed, y_train_usefulness, y_test_usefulness, best_params['usefulness'])
    print("Usefulness Prediction Report")
    print(usefulness_report)
