import pandas as pd
import numpy as np

# Load the datasets
req_df = pd.read_csv('../../dataset/all_requirements.csv', usecols=['id', 'uid'])
creativity_df = pd.read_csv('../../dataset/creativity-ratings.csv')


# Add user_id to creativity_df based on tid matching id in req_df
creativity_df['user_id'] = creativity_df['tid'].map(req_df.set_index('id')['uid'])

# Filter relevant columns and drop rows where user_id is NaN
creativity_df = creativity_df[['tid', 'user_id', 'detailedness', 'novelty', 'usefulness']].dropna(subset=['user_id'])

# Aggregate creativity scores by user_id
agg_functions = {'tid': lambda x: list(x), 'detailedness': 'mean', 'novelty': 'mean', 'usefulness': 'mean'}
creativity_df = creativity_df.groupby('user_id').aggregate(agg_functions).reset_index()

# Load the main dataset
df = pd.read_csv('../../cleaned_dataset/personblity_emotion_efficiency.csv')

# Add novelty and usefulness to the main dataframe　
df['Novelty'] = np.nan
df['Usefulness'] = np.nan
for user_id in creativity_df['user_id']:
    matching_user = creativity_df[creativity_df['user_id'] == user_id]
    if not matching_user.empty:
        df.loc[df['user_id'] == user_id, 'Novelty'] = matching_user['novelty'].values[0]
        df.loc[df['user_id'] == user_id, 'Usefulness'] = matching_user['usefulness'].values[0]

# Drop rows with NaN values
df = df.dropna()
df = df.reset_index(drop=True)

# Save the updated dataset
df.to_csv('../../cleaned_dataset/full_dataset4hypothesistest.csv', index=False)
