import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords

from ast import literal_eval

requirements = pd.read_csv('../../crowdre_question/requirements.csv')
req = pd.DataFrame(requirements, columns = ['feature','benefit','tags'])
req['sentence']= req['feature'] + ', ' + req['benefit']
req.drop(['feature','benefit'],axis=1, inplace=True)

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

req.dropna(inplace=True)
req.reset_index(drop=True, inplace=True)
req['tags'] = req.tags.apply(lambda x: list(re.split(r'\W+',x)))
req

len(req['tags'])



# generate first 30 most frequent tags
y = req['tags']

from nltk.stem import WordNetLemmatizer

interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','\'s']
STOPWORDS = set(stopwords.words('english'))

# generate first 30 most frequent tags

from nltk.stem import WordNetLemmatizer

# need to stemming the tag as well
tag_dic = {}

# counter for counting special cases
count = 0
count_air = 0
for tag_list, index in zip(req['tags'], range(len(req['tags']))):
    
    clean_list = []
    for tag in tag_list:

        # do lemmatization
        lemmatizer = WordNetLemmatizer()
        lemma_tag = lemmatizer.lemmatize(tag.lower(), 'n')

        # special case
        if lemma_tag == 'television':
            lemma_tag = 'tv'
            count += 1
        elif lemma_tag == 'conditioner' or lemma_tag == 'conditioning':
            lemma_tag = 'ac'
            count_air += 1
        # #######################
        # do stemming here
        stemmer = nltk.stem.SnowballStemmer('english')
        stem_tag = stemmer.stem(lemma_tag)
        # #######################

        if stem_tag not in (interpunctuations and STOPWORDS) and len(stem_tag) != 0:
            if stem_tag not in tag_dic:
                tag_dic[stem_tag] = 1
                clean_list.append(stem_tag)
            else:
                tag_dic[stem_tag] += 1
                clean_list.append(stem_tag)
                
    # replace with clean tag in req
    req['tags'][index] = clean_list


print('the time television has revised to tv:',count)  
print('the time air conditioner has revised to ac:',count_air)  
df = pd.DataFrame(list(tag_dic.items()), columns=['tag', 'count']).sort_values(by = 'count',axis = 0,ascending = False)
df.reset_index(drop=True, inplace=True)
print('the number of label type:',len(df))

total = 0

pd.set_option('display.max_rows', None)
#df.head(30)
df.head()

# Identify the tags that occur only once
tag_dic = {key: val for key, val in sorted(tag_dic.items(), key = lambda ele: ele[1], reverse = True)}     

low_occurrence_tags = [tag for tag, count in tag_dic.items() if count < 44]

print(len(low_occurrence_tags))
# Remove the tags from the DataFrame
for tag in low_occurrence_tags:
    #print(tag)
    for tags in req['tags']:
        if tag in tags:
            #print(tag)
            tags.remove(tag)
    if tag in tag_dic:
        tag_dic.pop(tag)

for tag in low_occurrence_tags:
    #print(tag)
    for tags in req['tags']:
        if tag in tags:
            #print(tag)
            tags.remove(tag)
            
del_tags = ['asthma', 'doorbel', 'updat', 'volum']
for tags in req['tags']:
    for tag in tags:
        if tag in del_tags:
            tags.remove(tag)

        
for tags, index in zip(req['tags'], range(len(req['tags']))):
    #print(len(tags))
    if len(tags)==0:
        req.drop(index, inplace=True)
        


            
#req = req[~req['tags'].apply(lambda x: tag in x)]

req.reset_index(drop=True, inplace=True)
req.head()

import numpy as np
import pandas as pd

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
            if train_tag_dic.get(tag, 0) < int(7/10*tag_dic[tag]):
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
            if np.random.uniform(0, 1) >= 1/2:
                train_tag_dic[tag] = train_tag_dic.get(tag, 0) + 1                
            else:
                t = 1
                test_tag_dic[tag] = test_tag_dic.get(tag, 0) + 1
    
    if t == 0:
        train.loc[index] = [tags, sentence]
    else:
        test.loc[index] = [tags, sentence]


train.reset_index(drop=True, inplace=True)

train.head()

test.reset_index(drop=True, inplace=True)

test.head()

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') 
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def text_prepare(text):
    text = text.lower() 
    text = REPLACE_BY_SPACE_RE.sub(' ',text) 
    text = BAD_SYMBOLS_RE.sub('',text)
    join_list = []
    for w in text.split():
        if w not in STOPWORDS:
            '''# do lemmatization
            lemmatizer = WordNetLemmatizer()
            lemma_w = lemmatizer.lemmatize(w, 'n')'''

            # special case
            if w == 'television':
                w = 'tv'

            '''# do stemming here
            stemmer = nltk.stem.SnowballStemmer('english')
            stem_w = stemmer.stem(lemma_w)
            
            join_list.append(stem_w)'''
            join_list.append(w)
            
            
            
    text = ' '.join(join_list)
    #text = ' '.join([w for w in text.split() if w not in STOPWORDS])
    return text

X_train, y_train = train.sentence, train.tags
X_test, y_test = test.sentence, test.tags
 
#data clean
X_train = [text_prepare(x) for x in X_train]
X_test = [text_prepare(x) for x in X_test]
X_train[:10]


cv = CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')
feature = cv.fit_transform(X_train)
print(feature.shape)
print(feature)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
 
def print_evaluation_scores(y_val, predicted):
    accuracy=accuracy_score(y_val, predicted)
    f1_score_macro=f1_score(y_val, predicted, average='macro')
    f1_score_micro=f1_score(y_val, predicted, average='micro')
    f1_score_weighted=f1_score(y_val, predicted, average='weighted')
    print("accuracy:",accuracy)
    print("f1_score_macro:",f1_score_macro)
    print("f1_score_micro:",f1_score_micro)
    print("f1_score_weighted:",f1_score_weighted)

print(type(y_train))
mlb = MultiLabelBinarizer(classes=sorted(tag_dic.keys()))
y_train = mlb.fit_transform(y_train)
y_test = mlb.fit_transform(y_test)
print(y_train.shape)
print(train.tags[0])
print(y_train[0])
print(mlb.classes_)

NB_pipeline = Pipeline([
                ('cv', CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
 
NB_pipeline.fit(X_train,y_train)
predicted_NB = NB_pipeline.predict(X_test)
print_evaluation_scores(y_test,predicted_NB)



from sklearn.metrics import multilabel_confusion_matrix

y_true = []
y_pred = []

if y_test.shape[0] == predicted_NB.shape[0]:
    for i in range(y_test.shape[0]):
        y_true.append(y_test[i])
        y_pred.append(predicted_NB[i])

y_true = np.array(y_test)
y_pred = np.array(predicted_NB)

# Assuming y_true and y_pred are the true and predicted labels, respectively
confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
classes =  ['alarm','alert','automat','child','clean','control','cook','door',
 'electr', 'energi', 'entertain', 'food', 'health' ,'heat' ,'home' ,'kitchen',
 'light', 'lock', 'music', 'pet', 'safeti', 'save', 'secur','sensor', 'shower',
 'smart', 'temperatur', 'tv' ,'water', 'window']# List of class labels

# Plot the confusion matrix for each label
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

SVC_pipeline = Pipeline([
                ('cv', CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
 
SVC_pipeline.fit(X_train,y_train)
predicted_SVC = SVC_pipeline.predict(X_test)
print_evaluation_scores(y_test,predicted_SVC)

LogReg_pipeline = Pipeline([
                ('cv', CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),
            ])
 
LogReg_pipeline.fit(X_train,y_train)
predicted_Log = LogReg_pipeline.predict(X_test)
print_evaluation_scores(y_test,predicted_Log)

knn_pipeline = Pipeline([
    ('cv', CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
    ('clf', OneVsRestClassifier(KNeighborsClassifier(), n_jobs=1))
])

# Fit the pipeline on training data
knn_pipeline.fit(X_train,y_train)

predicted_knn = knn_pipeline.predict(X_test)
print_evaluation_scores(y_test,predicted_knn)

tfidf = TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')
feature = tfidf.fit_transform(X_train)
print(feature.shape)
print(feature)

tfidf_NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(MultinomialNB())),
            ])
 
tfidf_NB_pipeline.fit(X_train,y_train)
tfidf_NB_predicted = tfidf_NB_pipeline.predict(X_test)
print_evaluation_scores(y_test,tfidf_NB_predicted)

tfidf_SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])

tfidf_SVC_pipeline.fit(X_train,y_train)
tfidf_SVC_predicted = tfidf_SVC_pipeline.predict(X_test)

print_evaluation_scores(y_test,tfidf_SVC_predicted)

tfidf_LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),
            ])
 
tfidf_LogReg_pipeline.fit(X_train,y_train)
tfidf_LogReg_predicted = tfidf_LogReg_pipeline.predict(X_test)
print_evaluation_scores(y_test,tfidf_LogReg_predicted)

tfidf_knn_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
    ('clf', OneVsRestClassifier(KNeighborsClassifier(), n_jobs=1))
])

# Fit the pipeline on training data
tfidf_knn_pipeline.fit(X_train,y_train)

tfidf_knn_predicted = tfidf_knn_pipeline.predict(X_test)
print_evaluation_scores(y_test,tfidf_knn_predicted)

# allocate tags for set 1
# set 1 data
scenarios = pd.read_csv('../../cleaned_dataset/senario.csv')
sc = pd.DataFrame(scenarios, columns = ['id','context','stimuli','response'])
sc['sentence']= sc['context'] + ', ' + sc['stimuli'] + ', ' +sc['response']
sc.drop(['context','stimuli','response'],axis=1, inplace=True)
sc.dropna(inplace=True)
sc.reset_index(drop=True, inplace=True)
sc.columns=['scenarios_id','sentence']
cv = sc.copy()
tf = sc.copy()
sc.head()

# clean set 1 sentence
XX_test = sc['sentence']
print(type(XX_test[0]))
XX_test[:10]

Xx_test = [text_prepare(x) for x in XX_test]
Xx_test[:10]

set1_predicted_SVC = SVC_pipeline.predict(Xx_test)
set1_predicted_SVC

set1_predicted_tfidf_LSVC = tfidf_knn_pipeline.predict(Xx_test)
set1_predicted_tfidf_LSVC

cv_tags = mlb.inverse_transform(set1_predicted_SVC)
cv['tags'] = cv_tags
cv.head()

tf_tags = mlb.inverse_transform(set1_predicted_tfidf_LSVC)
tf['tags'] = tf_tags
tf.head()

pre_count = 0
count = 0
for tags, index in zip(cv['tags'],range(len(cv['tags']))):
    if len(tags)==0:
        cv['tags'][index] = tf['tags'][index]
        pre_count += 1
    if len(cv['tags'][index])==0:
        count += 1
        
print('pre_count', pre_count)
print('count', count)
cv.head()
        
        

cv.set_index('scenarios_id',inplace=True)
cv.to_csv('predictions/predicted_tags.csv')





