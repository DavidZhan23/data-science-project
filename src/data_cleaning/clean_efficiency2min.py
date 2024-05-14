import pandas as pd
import numpy as np
import os
import re

# Load the dataset
df = pd.read_csv('../../cleaned_dataset/output0.csv')

def convert_to_minutes(time_str):
    # Normalize the string to lower case
    time_str = time_str.strip().lower()

    try:
        # Handle various formats
        if re.match(r'^\d+:\d+$', time_str):  # Format like 00:30 or 1:30
            parts = time_str.split(':')
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        elif re.match(r'^\d+\s?min', time_str) or re.match(r'^\d+\s?minutes', time_str):  # Format like 30 minutes
            return int(re.findall(r'\d+', time_str)[0])
        elif re.match(r'^\d+\s?hrs?', time_str):  # Format like 3 hrs
            return int(re.findall(r'\d+', time_str)[0]) * 60
        elif re.match(r'^\d+\s?hour', time_str):  # Format like 1 hour
            return int(re.findall(r'\d+', time_str)[0]) * 60
        elif re.match(r'^\d+:\d{2}', time_str):  # Format like 1:00, 00:60, etc.
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        elif re.match(r'^\d+\.?\d*$', time_str):  # Format like 30, 25, etc.
            return int(time_str.split('.')[0])
        elif 'about' in time_str or 'approximately' in time_str:  # Handle approximate times
            numbers = re.findall(r'\d+', time_str)
            if len(numbers) == 1:
                return int(numbers[0])
            elif len(numbers) == 2:
                return (int(numbers[0]) + int(numbers[1])) // 2
        elif re.match(r'\d+\s*:\s*\d+', time_str):  # Format like 00:30, 1:00, 1:30, etc.
            parts = re.findall(r'\d+', time_str)
            return int(parts[0]) * 60 + int(parts[1])
        else:
            print(f"Unrecognized format: {time_str}")
    except Exception as e:
        print(f"Error processing {time_str}: {e}")
    
    return np.nan

# Apply the conversion function to the Efficiency column
df['Efficiency'] = df['Efficiency'].apply(convert_to_minutes)

# Drop rows with NaN values
df = df.dropna()

# Create the cleaned_dataset folder if it doesn't exist
os.makedirs('../../cleaned_dataset', exist_ok=True)

# Save the cleaned dataframe
df.to_csv('../../cleaned_dataset/out32.csv', index=False)