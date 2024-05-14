import pandas as pd
import numpy as np
import os
import re

# Load the dataset
df = pd.read_csv('../../cleaned_dataset/output0.csv')


# Function to convert time strings to minutes
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
            parts = re.findall(r'\d+', time_str)
            return int(parts[0]) * 60 + int(parts[1])
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
        elif re.match(r'\d+\s*.\s*\d+', time_str):  # Format like 00.30, 1.00, 1.30, etc.
            parts = re.findall(r'\d+', time_str)
            return int(parts[0]) * 60 + int(parts[1])
        elif re.match(r'^\d+\s*:\s*\d+:\d+$', time_str):  # Format like 6:30:00
            parts = re.findall(r'\d+', time_str)
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) // 60
        elif re.match(r'\d+\s*-\s*\d+\s*minutes', time_str):  # Format like 45 - 60 minutes
            parts = re.findall(r'\d+', time_str)
            return (int(parts[0]) + int(parts[1])) // 2
        elif re.match(r'\d+\s*to\s*\d+\s*minutes', time_str):  # Format like 20 to 30 minutes
            parts = re.findall(r'\d+', time_str)
            return (int(parts[0]) + int(parts[1])) // 2
        elif re.match(r'\d+[:.]\d+', time_str):  # Format like 00.30 mins or 00:30
            parts = re.findall(r'\d+', time_str)
            return int(parts[0]) * 60 + int(parts[1])
        elif re.match(r'thirty minutes', time_str):  # Format like thirty minutes
            return 30
        elif re.match(r'twenty five minutes', time_str):  # Format like twenty five minutes
            return 25
        elif re.match(r'less than an hour', time_str):  # Format like less than an hour
            return 60
        elif re.match(r'over 10 hours', time_str):  # Format like can not view timer (over 10 hours)
            return 10 * 60
        elif re.match(r'about an hour', time_str):  # Format like about an hour
            return 60
        elif re.match(r'^:\d{2}', time_str):  # Format like :30, :40
            parts = re.findall(r'\d+', time_str)
            return int(parts[0])
        elif re.match(r'i didn\'t keep track of time, around \d+ minutes i think', time_str):  # Format like i didn't keep track of time, around 25 minutes i think
            parts = re.findall(r'\d+', time_str)
            return int(parts[0])
        elif re.match(r'can not view timer \(over \d+ hours\)', time_str):  # Format like can not view timer (over 10 hours)
            parts = re.findall(r'\d+', time_str)
            return int(parts[0]) * 60
        elif re.match(r'00.25', time_str):  # Format like about an hour
            return 60
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
