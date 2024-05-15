# This is to convert the time in the requirements.csv to rate. 
# We have used this code to gind out the rate and them plotted a plot for the rate
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# set figure text size to 25
plt.rcParams.update({'font.size': 40})

reqs_df = pd.read_csv('../../crowdre_question/requirements.csv')[['id', 'created_at']]
reqs_df['created_at'] = pd.to_datetime(reqs_df['created_at'])
ratings_df = pd.read_csv('../../crowdre_question/requirements_ratings.csv')[['requirement_id', 'created_at']]
ratings_df['created_at'] = pd.to_datetime(ratings_df['created_at'])
# rename column to rated at
ratings_df.rename(columns={'created_at': 'rated_at'}, inplace=True)
# merge reqs_df and ratings_df
df = pd.merge(reqs_df, ratings_df, left_on='id', right_on='requirement_id')
# drop requirement_id column
df.drop(columns=['requirement_id'], inplace=True)
# get time difference between created_at and rated_at
df['time_to_rate'] = df['rated_at'] - df['created_at']
# convert time_to_rate to days
df['time_to_rate'] = df['time_to_rate'].dt.days

# plot histogram of time_to_rate
plt.figure(figsize=(20, 12))
plt.hist(df['time_to_rate'], bins=50)
plt.xlabel('Time to rate (days)')
plt.ylabel('Number of requirements')
# save plot
plt.savefig('../../exploratory_plots/rate.png')
# print mean time_to_rate
print(f'Mean time to rate: {np.mean(df["time_to_rate"]):.0f} days')
# print min time_to_rate
print(f'Min time to rate: {np.min(df["time_to_rate"])} days')
# %%
