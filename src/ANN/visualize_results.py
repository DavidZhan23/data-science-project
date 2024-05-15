import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_preprocessing import load_and_preprocess_data

def train_rf_and_get_predictions(X_train, X_test, y_train, y_test, best_params):
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def save_predictions(y_test, y_pred, output_file):
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results.to_csv(output_file, index=False)

def plot_correlation_matrix(y_test, y_pred):
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    correlation_matrix = results.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Actual vs Predicted')
    plt.show()

if __name__ == "__main__":
    file_path = '../../cleaned_dataset/data4training.csv'

    with open("best_params.json", "r") as f:
        best_params = json.load(f)

    X_train_transformed, X_test_transformed, y_train_novelty, y_test_novelty, y_train_usefulness, y_test_usefulness = load_and_preprocess_data(file_path)

    # For novelty
    y_pred_novelty = train_rf_and_get_predictions(X_train_transformed, X_test_transformed, y_train_novelty, y_test_novelty, best_params['novelty'])
    save_predictions(y_test_novelty, y_pred_novelty, 'predictions/novelty_predictions.csv')
    plot_correlation_matrix(y_test_novelty, y_pred_novelty)

    # For usefulness
    y_pred_usefulness = train_rf_and_get_predictions(X_train_transformed, X_test_transformed, y_train_usefulness, y_test_usefulness, best_params['usefulness'])
    save_predictions(y_test_usefulness, y_pred_usefulness, 'predictions/usefulness_predictions.csv')
    plot_correlation_matrix(y_test_usefulness, y_pred_usefulness)
