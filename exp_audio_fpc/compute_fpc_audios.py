from tqdm import tqdm
import csv
import pandas as pd
from typing import List, Tuple
import numpy as np
import os

# Hypercube: [[1500, 8000], [0.015, 1.0], [0.15, 2], [10**-5, 0.3], [0.25, 1.0]]
min_omega, max_omega = 1500, 8000
min_log_omega, max_log_omega = np.log10(min_omega), np.log10(max_omega)
min_tau, max_tau = 0.015, 1.0
min_p, max_p = 0.15, 2
min_logp, max_logp = np.log10(min_p), np.log10(max_p)
min_D, max_D = 10**-5, 0.3
min_log_D, max_log_D = np.log10(min_D), np.log10(max_D)
min_alpha, max_alpha = 0.25, 1.0

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
    alpha_values = [0.01 * i for i in range(1, num_intermediate_samples)] + [0.99]

    # Load couples as torch tensor
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    trajectories = []
    for couple in tqdm(couples, desc=f"Computing FPNUC trajectories"):
        a, b = np.array(couple[0]), np.array(couple[1])

        # Initialize trajectory with point A
        trajectory = []
        trajectory.extend(a)

        for i in range(num_intermediate_samples):
            distance_from_a = (b-a) * alpha_values[i]
            intermediate_point = a + distance_from_a
            trajectory.extend(intermediate_point)

        trajectory.extend(b)
        assert len(trajectory) == (num_intermediate_samples + 2)*5, f"Expected {(num_intermediate_samples + 2)*5} points, got {len(trajectory)}"

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

    # Load couples as torch tensor
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    trajectories = []
    for couple in tqdm(couples, desc=f"Computing FPCC trajectories"):
        a, b = np.array(couple[0]), np.array(couple[1])

        # Initialize trajectory with point A
        trajectory = []
        trajectory.extend(a)

        # Generate intermediate points between A and B
        for i in range(1, num_intermediate_samples + 1):
            valid_parameters = False
            while not valid_parameters:
                # Random direction in the same shape as A
                random_direction = np.random.rand(len(a))
                random_direction = random_direction / np.linalg.norm(random_direction)  # Normalize

                # Distance from A proportional to progress
                distance_from_a = np.linalg.norm(a - b) * i / (num_intermediate_samples + 2)
                intermediate_point = a + distance_from_a * random_direction
                if (min_log_omega <= intermediate_point[0] <= max_log_omega and
                    min_tau <= intermediate_point[1] <= max_tau and
                    min_logp <= intermediate_point[2] <= max_logp and
                    min_log_D <= intermediate_point[3] <= max_log_D and
                    min_alpha <= intermediate_point[4] <= max_alpha):
                    valid_parameters = True

            trajectory.extend(intermediate_point)

        trajectory.extend(b)
        assert len(trajectory) == (num_intermediate_samples + 2)*5, f"Expected {(num_intermediate_samples + 2)*5} points, got {len(trajectory)}"

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