import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../../cleaned_dataset/output0.csv"
data = pd.read_csv(file_path)

# Convert Efficiency to a uniform time format (minutes) if necessary
def convert_time_to_minutes(time_str):
    try:
        if 'hour' in time_str or 'hr' in time_str:
            return int(time_str.split()[0]) * 60
        elif ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2])/60
        elif 'min' in time_str or 'mins' in time_str or 'minutes' in time_str:
            return int(time_str.split()[0])
        else:
            return int(time_str)
    except:
        return None

data['Efficiency'] = data['Efficiency'].apply(convert_time_to_minutes)

# Define the emotions and their corresponding columns
emotions = ["Enjoyment", "Boredom", "Confidence", "Anxiety"]

# Create a 2x2 grid for the plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plotting each emotion as a violin plot for different personality types
for i, emotion in enumerate(emotions):
    sns.violinplot(x='Personality', y=emotion, data=data, ax=axes[i])
    axes[i].set_title(f'Distribution of {emotion} by Personality Type')
    axes[i].set_xlabel('Personality Type')
    axes[i].set_ylabel(emotion)

plt.tight_layout()
plt.savefig('../../exploratory_plots/emotions_by_personality.png')
plt.show()
