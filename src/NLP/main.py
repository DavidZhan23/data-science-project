import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from data_processing import load_data, clean_tags, filter_tags, split_data, text_prepare
from model import build_and_evaluate_model, save_confusion_matrices, build_other_models, build_tfidf_models
from utils import download_nltk_data

def main():
    # download nltk
    download_nltk_data()

    # load my data
    req = load_data('../../crowdre_question/requirements.csv')

    # clean tags
    req, tag_dic = clean_tags(req)

    # filter tages
    req, tag_dic = filter_tags(req, tag_dic)

    # split the tags 
    train, test = split_data(req, tag_dic)

    X_train, y_train = train.sentence, train.tags
    X_test, y_test = test.sentence, test.tags

    X_train = [text_prepare(x) for x in X_train]
    X_test = [text_prepare(x) for x in X_test]

    # transform the tags
    mlb = MultiLabelBinarizer(classes=sorted(tag_dic.keys()))
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)

    # construct and evaluate the model here
    model, predicted = build_and_evaluate_model(X_train, y_train, X_test, y_test)

    # construct and save my confusion matrix here
    classes = ['alarm','alert','automat','child','clean','control','cook','door',
               'electr', 'energi', 'entertain', 'food', 'health' ,'heat' ,'home' ,'kitchen',
               'light', 'lock', 'music', 'pet', 'safeti', 'save', 'secur','sensor', 'shower',
               'smart', 'temperatur', 'tv' ,'water', 'window']
    save_confusion_matrices(y_test, predicted, classes)

    # evaluate other models
    # To make my terminel looks neater, I commented this part
    # But you are free to try any model and evaluate based on your onw preference.
    # You can go to build_other_models() method in model.py, there are comments guide you to do this
    build_other_models(X_train, y_train, X_test, y_test)

    # train a tfidf_model
    build_tfidf_models(X_train, y_train, X_test, y_test)

    # save theresult 
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
