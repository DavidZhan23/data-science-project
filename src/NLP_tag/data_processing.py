import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import numpy as np

STOPWORDS = set(stopwords.words('english'))

def load_data(filepath):
    requirements = pd.read_csv(filepath)
    req = pd.DataFrame(requirements, columns=['feature', 'benefit', 'tags'])
    req['sentence'] = req['feature'] + ', ' + req['benefit']
    req.drop(['feature', 'benefit'], axis=1, inplace=True)
    req.dropna(inplace=True)
    req.reset_index(drop=True, inplace=True)
    req['tags'] = req['tags'].apply(lambda x: list(re.split(r'\W+', x)))
    return req

def clean_tags(req):
    tag_dic = {}
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'s']
    count = 0
    count_air = 0
    for tag_list, index in zip(req['tags'], range(len(req['tags']))):
        clean_list = []
        for tag in tag_list:
            lemmatizer = WordNetLemmatizer()
            lemma_tag = lemmatizer.lemmatize(tag.lower(), 'n')

            if lemma_tag == 'television':
                lemma_tag = 'tv'
                count += 1
            elif lemma_tag == 'conditioner' or lemma_tag == 'conditioning':
                lemma_tag = 'ac'
                count_air += 1

            stemmer = SnowballStemmer('english')
            stem_tag = stemmer.stem(lemma_tag)

            if stem_tag not in (interpunctuations and STOPWORDS) and len(stem_tag) != 0:
                if stem_tag not in tag_dic:
                    tag_dic[stem_tag] = 1
                    clean_list.append(stem_tag)
                else:
                    tag_dic[stem_tag] += 1
                    clean_list.append(stem_tag)

        req.at[index, 'tags'] = clean_list

    print('the time television has revised to tv:', count)
    print('the time air conditioner has revised to ac:', count_air)
    
    return req, tag_dic

def filter_tags(req, tag_dic, min_count=44):
    low_occurrence_tags = [tag for tag, count in tag_dic.items() if count < min_count]
    for tag in low_occurrence_tags:
        for tags in req['tags']:
            if tag in tags:
                tags.remove(tag)
        if tag in tag_dic:
            tag_dic.pop(tag)

    for tag in ['asthma', 'doorbel', 'updat', 'volum']:
        for tags in req['tags']:
            if tag in tags:
                tags.remove(tag)

    for tags, index in zip(req['tags'], range(len(req['tags']))):
        if len(tags) == 0:
            req.drop(index, inplace=True)

    req.reset_index(drop=True, inplace=True)
    return req, tag_dic

def split_data(req, tag_dic):
    np.random.seed(0)
    train_tag_dic = {}
    test_tag_dic = {}

    train = pd.DataFrame(columns=['tags', 'sentence'])
    test = pd.DataFrame(columns=['tags', 'sentence'])

    index = 0
    for tags, sentence in zip(req['tags'], req['sentence']):
        index += 1
        zero_add_checker = 0
        add_checker = 0
        for tag in tags:
            if tag not in train_tag_dic:
                zero_add_checker = 1
                break

        if zero_add_checker == 0:
            check = 0
            for tag in tags:
                if train_tag_dic.get(tag, 0) < int(7 / 10 * tag_dic.get(tag, 0)):
                    check += 1

            if check == len(tags):
                add_checker = 1

        t = 0
        for tag in tags:
            if zero_add_checker == 1 or add_checker == 1:
                if tag not in train_tag_dic:
                    train_tag_dic[tag] = 1
                else:
                    train_tag_dic[tag] += 1

            elif tag not in test_tag_dic:
                t = 1
                test_tag_dic[tag] = 1

            else:
                if np.random.uniform(0, 1) >= 1 / 2:
                    train_tag_dic[tag] = train_tag_dic.get(tag, 0) + 1
                else:
                    t = 1
                    test_tag_dic[tag] = test_tag_dic.get(tag, 0) + 1

        if t == 0:
            train.loc[index] = [tags, sentence]
        else:
            test.loc[index] = [tags, sentence]

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return train, test

def text_prepare(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    join_list = []
    for w in text.split():
        if w not in STOPWORDS:
            if w == 'television':
                w = 'tv'
            join_list.append(w)
    text = ' '.join(join_list)
    return text
