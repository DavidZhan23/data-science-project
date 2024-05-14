# %%
import pandas as pd
import numpy as np
import os
# %%
users_df = pd.read_csv('../../dataset/users.csv')
users_df = users_df.sort_values(by=['id'])
survey_df = pd.read_csv('../../dataset/post-survey-responses.csv')
survey_df = survey_df.sort_values(by=['user_id'])
print(users_df.head())
# %%
# create a new dataframe with columns user_id, 1, 4, 5, 6, 7, personality
df = pd.DataFrame(columns=['user_id','Efficiency','Enjoyment','Boredom','Confidence','Anxiety','Personality','group_type'])

i = 0
for user_id in survey_df['user_id']:
    df.loc[i, 'user_id'] = user_id
    df.loc[i, 'Efficiency'] = survey_df.loc[(survey_df['user_id'] == user_id) & (survey_df['question_id'] == 1)]['description'].values[0]
    df.loc[i, 'Enjoyment'] = survey_df.loc[(survey_df['user_id'] == user_id) & (survey_df['question_id'] == 4)]['description'].values[0]
    df.loc[i, 'Boredom'] = survey_df.loc[(survey_df['user_id'] == user_id) & (survey_df['question_id'] == 5)]['description'].values[0]
    df.loc[i, 'Confidence'] = survey_df.loc[(survey_df['user_id'] == user_id) & (survey_df['question_id'] == 6)]['description'].values[0]
    df.loc[i, 'Anxiety'] = survey_df.loc[(survey_df['user_id'] == user_id) & (survey_df['question_id'] == 7)]['description'].values[0]
    df.loc[i, 'Personality'] = users_df.loc[users_df['id'] == user_id]['personality'].values[0]
    df.loc[i, 'group_type'] = users_df.loc[users_df['id'] == user_id]['group_type'].values[0]
    i += 1
# %%
# drop duplicates
df = df.drop_duplicates()
# drop rows with NaN values
df = df.dropna()
# %%
# Create the cleaned_dataset folder if it doesn't exist
os.makedirs('../../cleaned_dataset', exist_ok=True)

# Save the dataframe to csv in the cleaned_dataset folder
df.to_csv('../../cleaned_dataset/out32.csv', index=False)
# %%