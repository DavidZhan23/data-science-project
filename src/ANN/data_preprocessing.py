import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['tags'].fillna('', inplace=True)
    feature_columns = ['application_domain', 'tags']
    target_novelty = 'novelty'
    target_usefulness = 'usefulness'

    X_train, X_test, y_train_novelty, y_test_novelty = train_test_split(
        data[feature_columns], data[target_novelty], test_size=0.4, random_state=42)
    _, _, y_train_usefulness, y_test_usefulness = train_test_split(
        data[feature_columns], data[target_usefulness], test_size=0.4, random_state=42)

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
