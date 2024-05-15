import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

def print_evaluation_scores(y_val, predicted):
    accuracy = accuracy_score(y_val, predicted)
    f1_score_macro = f1_score(y_val, predicted, average='macro')
    f1_score_micro = f1_score(y_val, predicted, average='micro')
    f1_score_weighted = f1_score(y_val, predicted, average='weighted')
    print("accuracy:", accuracy)
    print("f1_score_macro:", f1_score_macro)
    print("f1_score_micro:", f1_score_micro)
    print("f1_score_weighted:", f1_score_weighted)

def plot_confusion_matrix(y_true, y_pred, classes, model_name, output_dir='confusion_matrix_application_domain'):
    conf_mat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(conf_mat, cmap='Blues')
    ax.set_title(f'Confusion matrix for {model_name}')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, conf_mat[i, j], ha='center', va='center', color='black')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close(fig)
