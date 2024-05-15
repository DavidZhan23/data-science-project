# In[1]:
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
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.corpus import stopwords

from ast import literal_eval

requirements = pd.read_csv('../../crowdre_question/requirements.csv')

req = pd.DataFrame(requirements, columns = ['feature','benefit','application_domain', 'application_domain_other'])
req['sentence']= req['feature'] + ', ' + req['benefit']
req.drop(['feature','benefit'],axis=1, inplace=True)

# In[2]:
req.head()

# In[3]:
req.info()

# In[4]:
req.application_domain.unique()

# In[5]:
for other, index in zip(req['application_domain_other'], range(len(req['application_domain_other']))):
    if type(other)!=float:
        req['application_domain_other'][index] = other.lower()
    
pd.set_option('display.max_rows', None)
req['application_domain_other'].value_counts()

# In[6]:
y = req['application_domain']
keep_list = ['health', 'energy', 'entertainment', 'safety']

for domain, index in zip(req['application_domain'], range(len(y))):
    
    if domain != 'Other':
        new_domain = domain.lower()
        req['application_domain'][index] = new_domain
    elif domain == 'Other':
        if req['application_domain_other'][index] in keep_list:
            req['application_domain'][index] = req['application_domain_other'][index]
        else:
            req.drop(index, inplace= True)

req.reset_index(drop=True, inplace=True)
req.drop(['application_domain_other'],axis=1, inplace=True)
req.head()

# In[7]:
req.application_domain.unique()

# In[8]:
train, test = train_test_split(req, test_size=0.3, random_state=10)
print(len(train))
print(len(test))

train.sample(10)

# In[9]:
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]') 
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

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
    return text

# In[10]:
X_train, y_train = train.sentence, train.application_domain
X_test, y_test = test.sentence, test.application_domain
 
#data clean
X_train = [text_prepare(x) for x in X_train]
X_test = [text_prepare(x) for x in X_test]
X_train[:10]

# In[None]:


# In[11]:
cv = CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')
feature = cv.fit_transform(X_train)
print(feature.shape)
print(feature)

# In[12]:
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

# In[13]:
#print(y_train.values[1])

# In[14]:
# from sklearn.preprocessing import OneHotEncoder
# classes = ['health', 'safety', 'entertainment', 'energy']
# encoder = OneHotEncoder(categories=[classes])

# y_train1 = []
# y_test1 = []
# for i in range(len(y_train.values)):
#     y_train1.append(y_train.values[i])
    
# for i in range(len(y_test)):
#     y_test1.append(y_test.values[i])

    
# y_train1 = np.array(y_train1).reshape(-1,1)
# y_test1 = np.array(y_test1).reshape(-1,1)

# y_train = encoder.fit_transform(y_train1).toarray()
# y_test = encoder.fit_transform(y_test1).toarray()
# #y_test = encoder.transform([[l] for l in y_test.values]).toarray()
# print(y_train)

# In[15]:
# encoder.categories_

# In[16]:
NB_pipeline = Pipeline([
                ('cv', CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
 
NB_pipeline.fit(X_train,y_train)
predicted_NB = NB_pipeline.predict(X_test)
print_evaluation_scores(y_test,predicted_NB)

# In[17]:
print(predicted_NB)

# In[18]:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming you have y_test and predicted_NB already defined

# Convert y_test and predicted_NB to arrays if they are not already
y_true = np.array(y_test)
y_pred = np.array(predicted_NB)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix as a heatmap
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(conf_mat, cmap='Blues')

# Set labels and ticks
classes = ['health', 'safety', 'entertainment', 'energy']
ax.set_title('Confusion matrix')
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Add text annotations
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, conf_mat[i, j], ha='center', va='center', color='black')

# Display the plot
plt.show()


# In[19]:
# draw confuson matrix for CV + NB: y_test
from sklearn.metrics import confusion_matrix

y_true = []
y_pred = []

if y_test.shape[0] == predicted_NB.shape[0]:
    for i in range(y_test.shape[0]):
        y_true.append(y_test.values[i])
        y_pred.append(predicted_NB[i])

y_true = np.array(y_test)
y_pred = np.array(predicted_NB)

# Compute the confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix as a heatmap
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(conf_mat, cmap='Blues')
ax.set_title('Confusion matrix')
ax.set_xticks(np.arange(len(conf_mat)))
ax.set_yticks(np.arange(len(conf_mat)))
ax.set_xticklabels(['health', 'safety', 'entertainment', 'energy'])
ax.set_yticklabels(['health', 'safety', 'entertainment', 'energy'])
for i in range(len(conf_mat)):
    for j in range(len(conf_mat)):
        ax.text(j, i, conf_mat[i, j], ha='center', va='center', color='black')
plt.show()

# In[20]:
SVC_pipeline = Pipeline([
                ('cv', CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
 
SVC_pipeline.fit(X_train,y_train)
predicted_SVC = SVC_pipeline.predict(X_test)
print_evaluation_scores(y_test,predicted_SVC)

# In[21]:
LogReg_pipeline = Pipeline([
                ('cv', CountVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),
            ])
 
LogReg_pipeline.fit(X_train,y_train)
predicted_Log = LogReg_pipeline.predict(X_test)
print_evaluation_scores(y_test,predicted_Log)

# In[22]:
tfidf_NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(MultinomialNB())),
            ])
 
tfidf_NB_pipeline.fit(X_train,y_train)
tfidf_NB_predicted = tfidf_NB_pipeline.predict(X_test)
print_evaluation_scores(y_test,tfidf_NB_predicted)

# In[23]:
tfidf_SVC_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])

tfidf_SVC_pipeline.fit(X_train,y_train)
tfidf_SVC_predicted = tfidf_SVC_pipeline.predict(X_test)

print_evaluation_scores(y_test,tfidf_SVC_predicted)

# In[24]:
tfidf_LogReg_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
                ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),
            ])
 
tfidf_LogReg_pipeline.fit(X_train,y_train)
tfidf_LogReg_predicted = tfidf_LogReg_pipeline.predict(X_test)
print_evaluation_scores(y_test,tfidf_LogReg_predicted)

# In[25]:
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

# In[26]:
# clean set 1 sentence
XX_test = sc['sentence']
print(type(XX_test[0]))
Xx_test = [text_prepare(x) for x in XX_test]
Xx_test[:10]

# In[27]:
set1_predicted_NB = NB_pipeline.predict(Xx_test)
set1_predicted_NB

# In[28]:
cv['domain'] = set1_predicted_NB
cv.head()

# In[29]:
cv.set_index('scenarios_id',inplace=True)
cv.to_csv('./predicted_domain4all_requirements.csv')
cv.head()



