import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import nltk

# 下载必要的NLTK数据
nltk.download('stopwords')
nltk.download('wordnet')

# 数据加载和预处理函数
def load_data(filepath):
    requirements = pd.read_csv(filepath)
    req = pd.DataFrame(requirements, columns=['feature', 'benefit', 'tags'])
    req['sentence'] = req['feature'] + ', ' + req['benefit']
    req.drop(['feature', 'benefit'], axis=1, inplace=True)
    req.dropna(inplace=True)
    req.reset_index(drop=True, inplace=True)
    req['tags'] = req['tags'].apply(lambda x: list(re.split(r'\\W+', x)))
    return req

# 准备数据集
def prepare_data(req):
    X = req['sentence']
    y = MultiLabelBinarizer().fit_transform(req['tags'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
def build_model():
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', OneVsRestClassifier(MultinomialNB()))
    ])
    return model

# 训练模型
def train_model(X_train, y_train):
    model = build_model()
    model.fit(X_train, y_train)
    return model

# 评估模型并保存混淆矩阵
def evaluate_model(model, X_test, y_test):
    predicted = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(f"Model accuracy: {score:.2f}")
    
    # 生成并保存混淆矩阵
    save_confusion_matrices(y_test, predicted)
    
    return predicted

# 保存混淆矩阵
def save_confusion_matrices(y_true, y_pred):
    confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
    classes =  ['alarm','alert','automat','child','clean','control','cook','door',
                'electr', 'energi', 'entertain', 'food', 'health' ,'heat' ,'home' ,'kitchen',
                'light', 'lock', 'music', 'pet', 'safeti', 'save', 'secur','sensor', 'shower',
                'smart', 'temperatur', 'tv' ,'water', 'window']  # 类标签列表
    
    # 创建保存混淆矩阵的目录
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

# 主函数
def main():
    # 加载和预处理数据
    data = load_data('../../crowdre_question/requirements.csv')
    
    # 拆分数据集
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # 评估模型并生成混淆矩阵
    predicted = evaluate_model(model, X_test, y_test)
    
    # 预测结果保存
    os.makedirs('predictions', exist_ok=True)
    np.save('predictions/predictions.npy', predicted)
    print("Predictions saved to 'predictions/predictions.npy'")
    
    # 生成并保存应用领域预测
    scenarios_data = load_data('../../cleaned_dataset/senario.csv')
    X_scenarios = scenarios_data['sentence']
    predicted_scenarios = model.predict(X_scenarios)
    scenarios_data['tags'] = [list(MultiLabelBinarizer().inverse_transform([tag])[0]) for tag in predicted_scenarios]
    scenarios_data.to_csv('predictions/predicted_tags.csv', index=False)
    print("Predicted tags saved to 'predictions/predicted_tags.csv'")

if __name__ == "__main__":
    main()
