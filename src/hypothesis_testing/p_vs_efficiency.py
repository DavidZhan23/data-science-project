import pandas as pd
import scipy.stats as stats

# Load data
data_path = '../../cleaned_dataset/personblity_emotion_efficiency.csv'
data = pd.read_csv(data_path)

# Perform hypothesis testing
results = []
groups = data['group_type'].unique()

for group in groups:
    results.append(f"Group {group}\n{'-'*40}\n")
    
    subset = data[data['group_type'] == group]
    
    eS = subset[subset['Personality'] == 'S']['Efficiency'].dropna().tolist()
    eC = subset[subset['Personality'] == 'C']['Efficiency'].dropna().tolist()
    eI = subset[subset['Personality'] == 'I']['Efficiency'].dropna().tolist()
    eD = subset[subset['Personality'] == 'D']['Efficiency'].dropna().tolist()

    # Perform Kruskal-Wallis H-test
    stat, p_val = stats.kruskal(eS, eC, eI, eD)
    
    # Determine significance
    if p_val < 0.1:
        result = f"Personality has a significant relation to efficiency in group {group} (p-value = {p_val:.5f})"
    else:
        result = f"Personality does not have a significant relation to efficiency in group {group} (p-value = {p_val:.5f})"
    
    print(result)
    results.append(result)

    results.append('\n')

# Save the results to a file
output_file_path = 'results/hypothesis_test_results_personality_vs_efficiency.txt'
with open(output_file_path, 'w') as f:
    for result in results:
        f.write(result + '\n')
