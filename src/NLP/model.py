import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def prepare_data(req):
    X = req['sentence']
    y = MultiLabelBinarizer().fit_transform(req['tags'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', OneVsRestClassifier(MultinomialNB()))
    ])
    return model

def train_model(X_train, y_train):
    model = build_model()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predicted = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score:.2f}")
    
    # Generate and save confusion matrices
    save_confusion_matrices(y_test, predicted)
    
    return predicted

def save_confusion_matrices(y_true, y_pred):
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    classes =  ['alarm','alert','automat','child','clean','control','cook','door',
                'electr', 'energi', 'entertain', 'food', 'health' ,'heat' ,'home' ,'kitchen',
                'light', 'lock', 'music', 'pet', 'safeti', 'save', 'secur','sensor', 'shower',
                'smart', 'temperatur', 'tv' ,'water', 'window']  # List of class labels
    
    # Create a directory to save the confusion matrices
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
        ax.set_yticklabels(['True','False'])
        
        plt.savefig(f'confusion_matrices/{label}_confusion_matrix.png')
        plt.close(fig)

if __name__ == "__main__":
    from data_processing import load_data
    data = load_data('requirements.csv')
    X_train, X_test, y_train, y_test = prepare_data(data)
    model = train_model(X_train, y_train)
    predicted = evaluate_model(model, X_test, y_test)
