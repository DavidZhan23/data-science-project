import json
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_and_preprocess_data

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [ 5, 10, 20],
        'min_samples_leaf': [2,4,6,8]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

if __name__ == "__main__":
    file_path = '../../cleaned_dataset/data4training.csv'
    X_train_transformed, _, y_train_novelty, _, y_train_usefulness, _ = load_and_preprocess_data(file_path)

    print("Tuning Random Forest for novelty...")
    best_params_novelty = tune_random_forest(X_train_transformed, y_train_novelty)
    print(f"Best parameters for novelty: {best_params_novelty}")

    print("Tuning Random Forest for usefulness...")
    best_params_usefulness = tune_random_forest(X_train_transformed, y_train_usefulness)
    print(f"Best parameters for usefulness: {best_params_usefulness}")

    # Save best parameters
    best_params = {
        'novelty': best_params_novelty,
        'usefulness': best_params_usefulness
    }
    
    with open("best_params.json", "w") as f:
        json.dump(best_params, f)
