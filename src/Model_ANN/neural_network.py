import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_transformed)
    X_test_scaled = scaler.transform(X_test_transformed)

    # Adjust labels to be zero-indexed
    y_train_novelty = y_train_novelty - 1
    y_test_novelty = y_test_novelty - 1
    y_train_usefulness = y_train_usefulness - 1
    y_test_usefulness = y_test_usefulness - 1

    return X_train_scaled, X_test_scaled, y_train_novelty, y_test_novelty, y_train_usefulness, y_test_usefulness

def build_and_train_nn(X_train, y_train, X_test, y_test, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred)
    print(report)

    return model

if __name__ == "__main__":
    file_path = '../../cleaned_dataset/data4training.csv'
    X_train, X_test, y_train_novelty, y_test_novelty, y_train_usefulness, y_test_usefulness = load_and_preprocess_data(file_path)

    print("Training Neural Network for novelty prediction...")
    novelty_model = build_and_train_nn(X_train, y_train_novelty, X_test, y_test_novelty, num_classes=5)

    print("Training Neural Network for usefulness prediction...")
    usefulness_model = build_and_train_nn(X_train, y_train_usefulness, X_test, y_test_usefulness, num_classes=5)

    novelty_model.save("trained_model/novelty_nn_model.h5")
    usefulness_model.save("trained_model/usefulness_nn_model.h5")