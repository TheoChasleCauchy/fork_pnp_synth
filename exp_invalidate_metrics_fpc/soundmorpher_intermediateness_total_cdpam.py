import re
import numpy as np
import csv
from tqdm import tqdm

def process_csv(input_path: str):
    
    trajectories = []
    with open(input_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            # Convert each value to float and group into tuples of 5 parameters
            points = [
                list(map(float, row[i:i+2]))
                for i in range(0, len(row), 2)
            ]
            trajectories.append(points)

    # Compute intermediateness value 
    intermediateness_values = []
    for trajectory in tqdm(trajectories, desc="Computing Intermediateness", total=len(trajectories)):
        intermediateness_value = sum([np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i])) for i in range(len(trajectory)-1)])
        intermediateness_values.append(intermediateness_value)

    return intermediateness_values

def compute_intermediateness_total_cdpam(points_csv: str, metric_csv: str):

    intermediateness_values = process_csv(points_csv)

    # Write intermediateness values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Intermediateness Total CDPAM"])
        for i, intermediateness_value in enumerate(intermediateness_values):
            writer.writerow([i, intermediateness_value])
        writer.writerow(["Mean Intermediateness", f"{np.mean(intermediateness_values)} +- {np.std(intermediateness_values)}"])