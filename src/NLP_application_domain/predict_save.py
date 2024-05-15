import pandas as pd

def predict_new_data(pipeline, X):
    return pipeline.predict(X)

def save_predictions(ids, sentences, predictions, file_path):
    results = pd.DataFrame({'id': ids, 'sentence': sentences, 'domain': predictions})
    results.set_index('id', inplace=True)
    results.to_csv(file_path)
