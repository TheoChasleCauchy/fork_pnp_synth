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

    # Compute smoothness value 
    smoothness_values = []
    for trajectory in tqdm(trajectories, desc="Computing Smoothness", total=len(trajectories)):
        dists = [np.linalg.norm(np.array(trajectory[i+1]) - np.array(trajectory[i])) for i in range(len(trajectory)-1)]
        smoothness_mean, smoothness_std = np.mean(dists), np.std(dists)
        smoothness_values.append((smoothness_mean, smoothness_std))

    return smoothness_values

def compute_smoothness_mean_cdpam(points_csv: str, metric_csv: str):

    smoothness_values = process_csv(points_csv)

    # Write smoothness values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Smoothness Mean CDPAM", "Smoothness Std CDPAM"])
        for i, (smoothness_mean, smoothness_std) in enumerate(smoothness_values):
            writer.writerow([i, smoothness_mean, smoothness_std])
        writer.writerow(["Mean Smoothness", f"{np.mean([mean for mean, _ in smoothness_values])} +- {np.std([mean for mean, _ in smoothness_values])}"])