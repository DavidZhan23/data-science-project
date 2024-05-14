import pandas as pd
import scipy.stats as stats

# Load the dataset
data_path = '../../cleaned_dataset/full_dataset4hypothesistest.csv'
data = pd.read_csv(data_path)

# Initialize results list
results = []

# Define the groups and emotions
groups = data['group_type'].unique()
emotions = ['Enjoyment', 'Boredom', 'Confidence', 'Anxiety']

# Perform hypothesis testing for each group and emotion
for group in groups:
    results.append(f"Group {group}\n----------------------------------------")
    group_data = data[data['group_type'] == group]
    
    for emotion in emotions:
        levels = group_data[emotion].unique()
        
        # Usefulness
        usefulness_data = [group_data[group_data[emotion] == level]['Usefulness'].dropna().tolist() for level in levels]
        stat_usefulness, p_val_usefulness = stats.kruskal(*usefulness_data)
        
        # Determine significance for Usefulness
        if p_val_usefulness < 0.05:
            result_usefulness = f"{emotion} influences Usefulness significantly (p-value = {p_val_usefulness:.5f})"
        else:
            result_usefulness = f"{emotion} does not influence Usefulness significantly (p-value = {p_val_usefulness:.5f})"
        
        print(result_usefulness)
        results.append(result_usefulness)
        
        # Novelty
        novelty_data = [group_data[group_data[emotion] == level]['Novelty'].dropna().tolist() for level in levels]
        stat_novelty, p_val_novelty = stats.kruskal(*novelty_data)
        
        # Determine significance for Novelty
        if p_val_novelty < 0.05:
            result_novelty = f"{emotion} influences Novelty significantly (p-value = {p_val_novelty:.5f})"
        else:
            result_novelty = f"{emotion} does not influence Novelty significantly (p-value = {p_val_novelty:.5f})"
        
        print(result_novelty)
        results.append(result_novelty)
    
    # Add a separator line between groups
    results.append("\n----------------------------------------\n")

# Combined section for all data
results.append("Combined Data\n----------------------------------------")
for emotion in emotions:
    levels = data[emotion].unique()
    
    # Usefulness
    usefulness_data = [data[data[emotion] == level]['Usefulness'].dropna().tolist() for level in levels]
    stat_usefulness, p_val_usefulness = stats.kruskal(*usefulness_data)
    
    # Determine significance for Usefulness
    if p_val_usefulness < 0.05:
        result_usefulness = f"{emotion} influences Usefulness significantly (p-value = {p_val_usefulness:.5f})"
    else:
        result_usefulness = f"{emotion} does not influence Usefulness significantly (p-value = {p_val_usefulness:.5f})"
    
    print(result_usefulness)
    results.append(result_usefulness)
    
    # Novelty
    novelty_data = [data[data[emotion] == level]['Novelty'].dropna().tolist() for level in levels]
    stat_novelty, p_val_novelty = stats.kruskal(*novelty_data)
    
    # Determine significance for Novelty
    if p_val_novelty < 0.05:
        result_novelty = f"{emotion} influences Novelty significantly (p-value = {p_val_novelty:.5f})"
    else:
        result_novelty = f"{emotion} does not influence Novelty significantly (p-value = {p_val_novelty:.5f})"
    
    print(result_novelty)
    results.append(result_novelty)

# Save the results to a file
output_file_path = 'results/hypothesis_test_results_emotion_vs_creativity.txt'
with open(output_file_path, 'w') as f:
    for result in results:
        f.write(result + '\n')
