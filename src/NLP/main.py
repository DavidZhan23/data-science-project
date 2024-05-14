import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from data_processing import load_data, clean_tags, filter_tags, split_data, text_prepare
from model import build_and_evaluate_model, save_confusion_matrices, build_other_models, build_tfidf_models
from utils import download_nltk_data

def main():
    # 下载nltk数据
    download_nltk_data()

    # 加载数据
    req = load_data('../../crowdre_question/requirements.csv')

    # 清理标签
    req, tag_dic = clean_tags(req)

    # 过滤标签
    req, tag_dic = filter_tags(req, tag_dic)

    # 拆分数据集
    train, test = split_data(req, tag_dic)

    X_train, y_train = train.sentence, train.tags
    X_test, y_test = test.sentence, test.tags

    X_train = [text_prepare(x) for x in X_train]
    X_test = [text_prepare(x) for x in X_test]

    # 转换标签
    mlb = MultiLabelBinarizer(classes=sorted(tag_dic.keys()))
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)

    # 构建和评估模型
    model, predicted = build_and_evaluate_model(X_train, y_train, X_test, y_test)

    # 保存混淆矩阵
    classes = ['alarm','alert','automat','child','clean','control','cook','door',
               'electr', 'energi', 'entertain', 'food', 'health' ,'heat' ,'home' ,'kitchen',
               'light', 'lock', 'music', 'pet', 'safeti', 'save', 'secur','sensor', 'shower',
               'smart', 'temperatur', 'tv' ,'water', 'window']
    save_confusion_matrices(y_test, predicted, classes)

    # 训练和评估其他模型
    build_other_models(X_train, y_train, X_test, y_test)

    # 训练和评估TFIDF模型
    build_tfidf_models(X_train, y_train, X_test, y_test)

    # 预测并保存结果
    scenarios = pd.read_csv('../../cleaned_dataset/senario.csv')
    sc = pd.DataFrame(scenarios, columns=['id', 'context', 'stimuli', 'response'])
    sc['sentence'] = sc['context'] + ', ' + sc['stimuli'] + ', ' + sc['response']
    sc.drop(['context', 'stimuli', 'response'], axis=1, inplace=True)
    sc.dropna(inplace=True)
    sc.reset_index(drop=True, inplace=True)
    sc.columns = ['scenarios_id', 'sentence']
    Xx_test = [text_prepare(x) for x in sc['sentence']]

    set1_predicted = model.predict(Xx_test)

    cv_tags = mlb.inverse_transform(set1_predicted)
    sc['tags'] = cv_tags

    sc.set_index('scenarios_id', inplace=True)
    sc.to_csv('predictions/predicted_tags.csv')

if __name__ == "__main__":
    main()
