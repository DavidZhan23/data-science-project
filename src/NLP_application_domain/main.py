import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from data_preprocessing import load_data, preprocess_application_domain
from data_cleaning import clean_data
from model_training import train_test_split_data, create_pipeline, train_model
from model_evaluation import print_evaluation_scores, plot_confusion_matrix
from predict_save import predict_new_data, save_predictions

# Load data
req = load_data('../../crowdre_question/requirements.csv')

# preprocess
req = preprocess_application_domain(req, keep_list=['health', 'energy', 'entertainment', 'safety'])

# Split data
train, test = train_test_split_data(req)
X_train, y_train = train.sentence, train.application_domain
X_test, y_test = test.sentence, test.application_domain

# Clean data
X_train = clean_data(X_train)
X_test = clean_data(X_test)

# Train and evaluate models
classifiers = [
    (MultinomialNB(), "MultinomialNB"),
    (LinearSVC(), "LinearSVC"),
    (LogisticRegression(), "LogisticRegression")
]

for clf, clf_name in classifiers:
    pipeline = create_pipeline(clf)
    pipeline = train_model(pipeline, X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(f"Evaluation for {clf_name}:")
    print_evaluation_scores(y_test, predictions)
    plot_confusion_matrix(y_test, predictions, classes=['health', 'safety', 'entertainment', 'energy'], model_name=clf_name)

# Predict new data
scenarios = pd.read_csv('../../cleaned_dataset/senario.csv')
scenarios['sentence'] = scenarios['context'] + ', ' + scenarios['stimuli'] + ', ' + scenarios['response']
scenarios.dropna(subset=['sentence'], inplace=True)  # Remove rows with NaN in 'sentence'
X_new = clean_data(scenarios['sentence'])
predictions = predict_new_data(pipeline, X_new)
save_predictions(scenarios['id'], scenarios['sentence'], predictions, './predicted_domain4all_requirements.csv')