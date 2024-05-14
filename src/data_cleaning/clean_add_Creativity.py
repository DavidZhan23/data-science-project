# %%
import pandas as pd
import numpy as np
# %%
req_df = pd.read_csv('../../dataset/all_requirements.csv', usecols=['id', 'uid'])
creativity_df = pd.read_csv('../../dataset/creativity-ratings.csv')
# %%
i = 0
for id in creativity_df['tid']:
    creativity_df.loc[i, 'user_id'] = req_df.loc[req_df['id'] == id, 'uid'].values[0]
    i += 1
creativity_df = creativity_df[['tid', 'user_id', 'detailedness', 'novelty', 'usefulness']]
# %%
agg_functions = {'tid': lambda x: list(x), 'detailedness': 'mean', 'novelty': 'mean', 'usefulness': 'mean'}
creativity_df = creativity_df.groupby('user_id').aggregate(agg_functions)
creativity_df = creativity_df.reset_index()
# %%
df = pd.read_csv('../../cleaned_dataset/personblity_emotion_efficiency.csv')
# %%
df['Novelty'] = np.nan
df['Usefulness'] = np.nan
for i, user_id in enumerate(df['user_id']):
    if user_id in creativity_df['user_id'].values:
        df.loc[i, 'Novelty'] = creativity_df.loc[creativity_df['user_id'] == user_id, 'novelty'].values[0]
        df.loc[i, 'Usefulness'] = creativity_df.loc[creativity_df['user_id'] == user_id, 'usefulness'].values[0]
df = df.dropna()
df = df.reset_index(drop=True)
# %%
df.to_csv('../../cleaned_dataset/full_dataset4hypothesistest.csv', index=False)
# %%
