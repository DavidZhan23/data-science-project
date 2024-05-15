import pandas as pd
import scipy.stats as stats

# Load data
data_path = '../../cleaned_dataset/personblity_emotion_efficiency.csv'
data = pd.read_csv(data_path)

# Perform hypothesis testing
results = []
emotions_to_test = ['Enjoyment', 'Boredom', 'Confidence', 'Anxiety']
groups = data['group_type'].unique()

for group in groups:
    results.append(f"Group {group}\n{'-'*40}\n")
    
    for emotion in emotions_to_test:
        personalityS = data[(data['group_type'] == group) & (data['Personality'] == 'S')][emotion].dropna().tolist()
        personalityC = data[(data['group_type'] == group) & (data['Personality'] == 'C')][emotion].dropna().tolist()
        personalityI = data[(data['group_type'] == group) & (data['Personality'] == 'I')][emotion].dropna().tolist()
        personalityD = data[(data['group_type'] == group) & (data['Personality'] == 'D')][emotion].dropna().tolist()

        # Perform Kruskal-Wallis H-test
        stat, p_val = stats.kruskal(personalityS, personalityC, personalityI, personalityD)
        
        # Determine significance
        if p_val < 0.1:
            result = f"The influence of personality on {emotion} is significant (p-value = {p_val:.5f})"
        else:
            result = f"The influence of personality on {emotion} is not significant (p-value = {p_val:.5f})"
        
        print(result)
        results.append(result)

    results.append('\n')

# Save the results to a file
output_file_path = 'results/hypothesis_test_results_personality_vs_emotion.txt'
with open(output_file_path, 'w') as f:
    for result in results:
        f.write(result + '\n')
