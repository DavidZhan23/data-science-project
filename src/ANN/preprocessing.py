# %%
# preprocess for ANN, save the preprocessed data to cleaned_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import re
# %%
# get current working directory
path = f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/dataset/set2/'
# read requirements from set2
requirements_df = pd.read_csv('../../crowdre_question/requirements.csv')[
    ['id', 'user_id', 'application_domain','application_domain_other', 'tags']].sort_values(by=['id'])
# read ratings from set2
ratings_df = pd.read_csv('../../crowdre_question/requirements_ratings.csv')[
    ['requirement_id', 'novelty', 'usefulness','clarity']].sort_values(by=['requirement_id'])


requirements_df = requirements_df[requirements_df['application_domain'] != 'Other']
requirements_df = requirements_df.reset_index(drop=True)
# concat the two dataframe
df = pd.merge(requirements_df, ratings_df, left_on='id', right_on='requirement_id')
df = df.drop(columns=['requirement_id','application_domain_other'])
df.rename(columns={'id':'requirement_id'}, inplace=True)
# save df to csv
df.to_csv('../../cleaned_dataset/data4training.csv', index=False)
# %%