# This script converts the time information in requirements.csv to a rate metric.
# We used this code to determine the rate and subsequently plotted a graph for it.
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set the figure text size to 40
plt.rcParams.update({'font.size': 40})

# Load the data from CSV files
requirements_df = pd.read_csv('../../crowdre_question/requirements.csv')[['id', 'created_at']]
requirements_df['created_at'] = pd.to_datetime(requirements_df['created_at'])

ratings_df = pd.read_csv('../../crowdre_question/requirements_ratings.csv')[['requirement_id', 'created_at']]
ratings_df['created_at'] = pd.to_datetime(ratings_df['created_at'])

# Rename 'created_at' column to 'rated_at' in the ratings dataframe
ratings_df.rename(columns={'created_at': 'rated_at'}, inplace=True)

# Merge the requirements and ratings dataframes on the relevant keys
merged_df = pd.merge(requirements_df, ratings_df, left_on='id', right_on='requirement_id')

# Remove the 'requirement_id' column as it is no longer needed
merged_df.drop(columns=['requirement_id'], inplace=True)

# Calculate the time difference between creation and rating dates
merged_df['rating_time'] = merged_df['rated_at'] - merged_df['created_at']
# %%

# Convert the time difference to days
merged_df['rating_time'] = merged_df['rating_time'].dt.days

# Plot a histogram of the time taken to rate
plt.figure(figsize=(20, 12))
plt.hist(merged_df['rating_time'], bins=50)
plt.xlabel('Time to Rate (days)')
plt.ylabel('Number of Requirements')

# Save the plot as a PNG file
plt.savefig('../../exploratory_plots/rate.png')

