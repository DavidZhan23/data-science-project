# The models are defined here. You are also encouraged to adjust any model in your own interest. 
# I have implemented the method print_evaluation_scores() to review the performance of the model
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix


def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("accuracy:", accuracy)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)

def build_and_evaluate_model(X_train, y_train, X_test, y_test, vectorizer=CountVectorizer):
    pipeline = Pipeline([
        ('vectorizer', vectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
        ('clf', OneVsRestClassifier(MultinomialNB()))
    ])

    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)
    print_evaluation_scores(y_test, predicted)
    return pipeline, predicted

def save_confusion_matrices(y_true, y_pred, classes):
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)

    os.makedirs('confusion_matrices', exist_ok=True)

    for i, label in enumerate(classes):
        fig, ax = plt.subplots()
        mat = confusion_matrices[i]
        ax.imshow(mat, cmap='Blues')
        for (j, k), z in np.ndenumerate(mat):
            ax.text(k, j, '{:0.1f}'.format(z), ha='center', va='center')

        ax.set_title('Confusion matrix for ' + label)
        ax.set_xticks(np.arange(len(mat)))
        ax.set_yticks(np.arange(len(mat)))
        ax.set_xticklabels(['Positive', 'Negative'])
        ax.set_yticklabels(['True', 'False'])

        plt.savefig(f'confusion_matrices/{label}_confusion_matrix.png')
        plt.close(fig)



# below are alternative models can be used. 
# Further more, it is flaxible for you to choose any model you would like to, the reviewer can construct any model in your own interest 
# to see its performance 

def build_other_models(X_train, y_train, X_test, y_test):
    models = [
        ('SVC', LinearSVC()),
        ('LogReg', LogisticRegression()),
        ('KNN', KNeighborsClassifier())
    ]

    for name, clf in models:
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
            ('clf', OneVsRestClassifier(clf, n_jobs=1)),
        ])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        # print(f"Results for {name}:")
        # print_evaluation_scores(y_test, predicted)

def build_tfidf_models(X_train, y_train, X_test, y_test):
    models = [
        ('tfidf_NB', MultinomialNB()),
        ('tfidf_SVC', LinearSVC()),
        ('tfidf_LogReg', LogisticRegression()),
        ('tfidf_KNN', KNeighborsClassifier())
    ]

    for name, clf in models:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),
            ('clf', OneVsRestClassifier(clf, n_jobs=1)),
        ])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        # print(f"Results for {name}:")
        # print_evaluation_scores(y_test, predicted)
