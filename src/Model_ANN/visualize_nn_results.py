import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from data_preprocessing import load_and_preprocess_data

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    file_path = '../../cleaned_dataset/data4training.csv'

    X_train, X_test, y_train_novelty, y_test_novelty, y_train_usefulness, y_test_usefulness = load_and_preprocess_data(file_path)

    # Load trained models
    novelty_model = load_model("trained_model/novelty_nn_model.h5")
    usefulness_model = load_model("trained_model/usefulness_nn_model.h5")

    # Make predictions
    y_pred_novelty = np.argmax(novelty_model.predict(X_test), axis=1)
    y_pred_usefulness = np.argmax(usefulness_model.predict(X_test), axis=1)

    # Plot confusion matrices
    labels = ['1', '2', '3', '4', '5']
    
    print("Confusion Matrix for Novelty")
    plot_confusion_matrix(y_test_novelty, y_pred_novelty, labels, title='Confusion Matrix for Novelty')

    print("Confusion Matrix for Usefulness")
    plot_confusion_matrix(y_test_usefulness, y_pred_usefulness, labels, title='Confusion Matrix for Usefulness')
