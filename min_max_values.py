import pandas as pd

# Load the CSV file
df = pd.read_csv('icassp23/data/full_param_log.csv')

# Select parameter columns (excluding ID and fold)
param_columns = ['omega', 'tau', 'p', 'D', 'alpha']

# Calculate statistics
stats = df[param_columns].agg(['min', 'max', 'mean']).T

print("Statistics for each parameter column:")
print(stats)