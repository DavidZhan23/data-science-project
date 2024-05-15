from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def train_test_split_data(req, test_size=0.3, random_state=10):
    train, test = train_test_split(req, test_size=test_size, random_state=random_state)
    return train, test

def create_pipeline(classifier):
    pipeline = Pipeline([
        ('cv', CountVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
        ('clf', OneVsRestClassifier(classifier)),
    ])
    return pipeline

def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline
