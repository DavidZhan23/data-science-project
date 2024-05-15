# This is seperately created for tuning the hyperparameter of the random forest model 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['tags'].fillna('', inplace=True)
    feature_columns = ['application_domain', 'tags']
    target_novelty = 'novelty'
    target_usefulness = 'usefulness'

    X_train, X_test, y_train_novelty, y_test_novelty = train_test_split(
        data[feature_columns], data[target_novelty], test_size=0.2, random_state=42)
    _, _, y_train_usefulness, y_test_usefulness = train_test_split(
        data[feature_columns], data[target_usefulness], test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('app_domain', OneHotEncoder(), ['application_domain']),
            ('tags', TfidfVectorizer(), 'tags')
        ])

    preprocessor_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    X_train_transformed = preprocessor_pipeline.fit_transform(X_train)
    X_test_transformed = preprocessor_pipeline.transform(X_test)

    return X_train_transformed, X_test_transformed, y_train_novelty, y_test_novelty, y_train_usefulness, y_test_usefulness

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

if __name__ == "__main__":
    file_path = 'data4training.csv'
    X_train_transformed, _, y_train_novelty, _, y_train_usefulness, _ = load_and_preprocess_data(file_path)

    print("Tuning Random Forest for novelty...")
    best_params_novelty = tune_random_forest(X_train_transformed, y_train_novelty)
    print(f"Best parameters for novelty: {best_params_novelty}")

    print("Tuning Random Forest for usefulness...")
    best_params_usefulness = tune_random_forest(X_train_transformed, y_train_usefulness)
    print(f"Best parameters for usefulness: {best_params_usefulness}")

    # Save best parameters
    with open("best_params.txt", "w") as f:
        f.write(f"Novelty: {best_params_novelty}\n")
        f.write(f"Usefulness: {best_params_usefulness}\n")
