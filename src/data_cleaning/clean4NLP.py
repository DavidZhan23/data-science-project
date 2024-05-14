# %%
import pandas as pd
import numpy as np
import os

# %%

df = pd.read_csv('../../dataset/all_requirements.csv', usecols=['id', 'context', 'stimuli' , 'response', 'created_at'])




# %%
df.to_csv('../../cleaned_dataset/senario.csv', index=False)
# %%
