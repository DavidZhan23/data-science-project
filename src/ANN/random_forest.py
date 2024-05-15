#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv('../../cleaned_dataset/data4training.csv')

#%%
data['tags'].fillna('', inplace=True)

# Define feature columns and target columns
feature_columns = ['application_domain', 'tags']
target_novelty = 'novelty'
target_usefulness = 'usefulness'

# Split the data into training and testing sets
X_train, X_test, y_train_novelty, y_test_novelty = train_test_split(
    data[feature_columns], data[target_novelty], test_size=0.2, random_state=42)

_, _, y_train_usefulness, y_test_usefulness = train_test_split(
    data[feature_columns], data[target_usefulness], test_size=0.2, random_state=42)

# Preprocess 'application_domain' using OneHotEncoding and 'tags' using TF-IDF
preprocessor = ColumnTransformer(
    transformers=[
        ('app_domain', OneHotEncoder(), ['application_domain']),
        ('tags', TfidfVectorizer(), 'tags')
    ])

# Preprocessing pipeline
preprocessor_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Transform the features
X_train_transformed = preprocessor_pipeline.fit_transform(X_train)
X_test_transformed = preprocessor_pipeline.transform(X_test)

# Train a model to predict 'novelty'
model_novelty = LogisticRegression(max_iter=1000, random_state=42)
model_novelty.fit(X_train_transformed, y_train_novelty)

# Predict 'novelty' on the test set
y_pred_novelty = model_novelty.predict(X_test_transformed)

# Evaluate the model
novelty_report = classification_report(y_test_novelty, y_pred_novelty, output_dict=True)
print("Novelty Prediction Report")
print(novelty_report)

# Train a model to predict 'usefulness'
model_usefulness = LogisticRegression(max_iter=1000, random_state=42)
model_usefulness.fit(X_train_transformed, y_train_usefulness)

# Predict 'usefulness' on the test set
y_pred_usefulness = model_usefulness.predict(X_test_transformed)

# Evaluate the model
usefulness_report = classification_report(y_test_usefulness, y_pred_usefulness, output_dict=True)
print("Usefulness Prediction Report")
print(usefulness_report)