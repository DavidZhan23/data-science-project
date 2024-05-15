import pandas as pd
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the requirements CSV
requirements = pd.read_csv('../../crowdre_question/requirements.csv')
req = pd.DataFrame(requirements, columns=['feature', 'benefit', 'tags'])
req['sentence'] = req['feature'] + requirements['benefit']
req.drop(['feature', 'benefit'], axis=1, inplace=True)

# Fill NaN tags with 'other' and split tags into a list
req["tags"] = req["tags"].fillna('other')
req['tags'] = req.tags.apply(lambda x: re.split(', |,|#|; ', x))

# Split the dataset into train and test
train, test = train_test_split(req, test_size=0.3, random_state=10)
print(f"Train size: {len(train)}, Test size: {len(test)}")

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = nltk.stem.SnowballStemmer('english')
interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'s']

# Initialize tag dictionary and special case counter
tag_dic = {}
count = 0

# Clean tags
for tag_list, index in zip(req['tags'], range(len(req['tags']))):
    clean_list = []
    for tag in tag_list:
        if isinstance(tag, float):
            tag = 'other'
        
        # Lemmatize and stem the tag
        lemma_tag = lemmatizer.lemmatize(tag, 'n')
        if lemma_tag == 'television':
            lemma_tag = 'tv'
            count += 1
        
        stem_tag = stemmer.stem(lemma_tag)
        
        if stem_tag not in interpunctuations and len(stem_tag) != 0:
            if stem_tag not in tag_dic:
                tag_dic[stem_tag] = 1
                clean_list.append(stem_tag)
            else:
                tag_dic[stem_tag] += 1
                clean_list.append(stem_tag)
                
    # Replace with clean tags in req
    req['tags'][index] = clean_list

print('Number of times "television" was revised to "tv":', count)                

# Create a DataFrame for tags and their counts
df = pd.DataFrame(list(tag_dic.items()), columns=['tag', 'count']).sort_values(by='count', axis=0, ascending=False)
df.reset_index(drop=True, inplace=True)
print('Number of unique tags:', len(df))

# Plot and save the distribution of all tags
plt.figure(figsize=(10, 6))
sns.barplot(x=df['tag'][:30], y=df['count'][:30])
plt.title('Top 30 Tag Distribution', fontsize=18)
plt.xlabel('Tags', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../../exploratory_plots/tag_distribution.png')
plt.show()

# Plot and save the top 20 most frequent tags
plt.figure(figsize=(10, 6))
sns.barplot(x=df['tag'][:20], y=df['count'][:20])
plt.title('Top 20 Most Frequent Tags', fontsize=18)
plt.xlabel('Tags', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('../../exploratory_plots/top_20_tags.png')
plt.show()


tagCount=train['tags'].apply(lambda x : len(x))
x = tagCount.value_counts()
#plot label distribution 
plt.figure(figsize=(8,5))
plt.plot(x.index, x)
plt.title("label distribution",fontsize=15)
plt.ylabel('count', fontsize=15)
plt.xlabel('number of labels', fontsize=15)

plt.savefig('../../exploratory_plots/label_distribution.png')
plt.show()