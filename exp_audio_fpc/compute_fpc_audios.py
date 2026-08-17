from tqdm import tqdm
import csv
import pandas as pd
from typing import List, Tuple
import numpy as np
import os

def load_and_extract_couples(csv_file_path: str) -> List[Tuple[List[float], List[float]]]:
    # Load the CSV file
    df = pd.read_csv(csv_file_path, header=None)

    # Skip header
    df = df.iloc[1:]

    # Extract the couples (A, B) as tuples of lists
    couples = []
    for _, row in df.iterrows():
        A = [float(x) for x in row[:5].tolist()]  # First 5 elements as vector A
        B = [float(x) for x in row[5:10].tolist()]  # Next 5 elements as vector B
        couples.append((A, B))

    return couples


def create_fpnuc_intermediate_points(num_intermediate_samples, dirname):

    # Load couples as torch tensor
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    trajectories = []
    for couple in tqdm(couples, desc=f"Computing FPNUC trajectories"):
        a, b = couple[0], couple[1]

        # Initialize trajectory with point A
        trajectory = []
        trajectory.extend(a)

        # Generate intermediate points between A and B
        for i in range(1, num_intermediate_samples + 1):
            # Random direction in the same shape as A
            random_direction = np.random.rand(len(a))
            random_direction = random_direction / np.linalg.norm(random_direction)  # Normalize

            # Distance from A proportional to progress
            distance_from_a = np.linalg.norm(a - b) * i / (num_intermediate_samples + 2)
            intermediate_point = a + distance_from_a * random_direction

            trajectory.append(intermediate_point)

        trajectory.extend(b)
        assert len(trajectory) == num_intermediate_samples + 2

        trajectories.append(trajectory)

    # Save to CSV
    os.makedirs(dirname, exist_ok=True)
    filepath = os.path.join(dirname, "fpnuc_trajectories.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filepath


def create_fpcc_intermediate_points(num_intermediate_samples, dirname):
    alpha_values = [0.01 * i for i in range(1, num_intermediate_samples)] + [0.99]

    # Load couples as torch tensor
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    trajectories = []
    for couple in tqdm(couples, desc=f"Computing FPCC trajectories"):
        a, b = couple[0], couple[1]

        # Initialize trajectory with point A
        trajectory = []
        trajectory.extend(a)

        for i in range(num_intermediate_samples):
            distance_from_a = (b-a) * alpha_values[i]
            intermediate_point = a + distance_from_a
            trajectory.append(intermediate_point)

        trajectory.extend(b)
        assert len(trajectory) == num_intermediate_samples + 2

        trajectories.append(trajectory)

    # Save to CSV
    os.makedirs(dirname, exist_ok=True)
    filepath = os.path.join(dirname, "fpcc_trajectories.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filepath