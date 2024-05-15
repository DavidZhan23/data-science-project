import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from data_preprocessing import load_and_preprocess_data

def train_rf_and_get_predictions(X_train, X_test, y_train, y_test, best_params):
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def save_predictions(y_test, y_pred, output_file):
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results.to_csv(output_file, index=False)

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    file_path = '../../cleaned_dataset/data4training.csv'
    output_dir = 'confusion_matrix'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open("best_params.json", "r") as f:
        best_params = json.load(f)

    X_train_transformed, X_test_transformed, y_train_novelty, y_test_novelty, y_train_usefulness, y_test_usefulness = load_and_preprocess_data(file_path)

    # For novelty
    y_pred_novelty = train_rf_and_get_predictions(X_train_transformed, X_test_transformed, y_train_novelty, y_test_novelty, best_params['novelty'])
    save_predictions(y_test_novelty, y_pred_novelty, os.path.join(output_dir, 'novelty_predictions.csv'))
    plot_confusion_matrix(y_test_novelty, y_pred_novelty, labels=['1', '2', '3', '4', '5'], 
                          title='Confusion Matrix for Novelty', 
                          filename=os.path.join(output_dir, 'confusion_matrix_novelty_rf.png'))

    # For usefulness
    y_pred_usefulness = train_rf_and_get_predictions(X_train_transformed, X_test_transformed, y_train_usefulness, y_test_usefulness, best_params['usefulness'])
    save_predictions(y_test_usefulness, y_pred_usefulness, os.path.join(output_dir, 'usefulness_predictions.csv'))
    plot_confusion_matrix(y_test_usefulness, y_pred_usefulness, labels=['1', '2', '3', '4', '5'], 
                          title='Confusion Matrix for Usefulness', 
                          filename=os.path.join(output_dir, 'confusion_matrix_usefulness_rf.png'))
