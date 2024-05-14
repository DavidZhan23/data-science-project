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
    
    subset = data[data['group_type'] == group]
    
    for emotion in emotions_to_test:
        efficiency_1 = subset[subset[emotion] == 1]['Efficiency'].dropna().tolist()
        efficiency_2 = subset[subset[emotion] == 2]['Efficiency'].dropna().tolist()
        efficiency_3 = subset[subset[emotion] == 3]['Efficiency'].dropna().tolist()
        efficiency_4 = subset[subset[emotion] == 4]['Efficiency'].dropna().tolist()
        efficiency_5 = subset[subset[emotion] == 5]['Efficiency'].dropna().tolist()
        
        # Perform Kruskal-Wallis H-test
        stat, p_val = stats.kruskal(efficiency_1, efficiency_2, efficiency_3, efficiency_4, efficiency_5)
        
        # Determine significance
        if p_val < 0.05:
            result = f"{emotion} influences efficiency significantly (p-value = {p_val:.5f})"
        else:
            result = f"{emotion} does not influence efficiency significantly (p-value = {p_val:.5f})"
        
        print(result)
        results.append(result)

    results.append('\n')

# Save the results to a file
output_file_path = 'results/hypothesis_test_results_emotion_vs_efficiency.txt'
with open(output_file_path, 'w') as f:
    for result in results:
        f.write(result + '\n')
