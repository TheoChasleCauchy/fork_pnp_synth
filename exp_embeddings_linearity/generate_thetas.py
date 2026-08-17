import os
import random
import csv
import numpy as np

# Hypercube: [[1500, 8000], [0.015, 1.0], [0.15, 2], [10**-5, 0.3], [0.25, 1.0]]
min_omega, max_omega = 1500, 8000
min_log_omega, max_log_omega = np.log10(min_omega), np.log10(max_omega)
min_tau, max_tau = 0.015, 1.0
min_p, max_p = 0.15, 2
min_logp, max_logp = np.log10(min_p), np.log10(max_p)
min_D, max_D = 10**-5, 0.3
min_log_D, max_log_D = np.log10(min_D), np.log10(max_D)
min_alpha, max_alpha = 0.25, 1.0

def generate_and_save_couples(seed, number_of_couples, filename="exp_embeddings_linearity/generated/thetas_couples.csv"):

    random.seed(seed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    couples = []

    for _ in range(number_of_couples):
        log_omega = random.uniform(min_log_omega, max_log_omega)
        tau = random.uniform(min_tau, max_tau)
        log_D = random.uniform(min_log_D, max_log_D)
        alpha = random.uniform(min_alpha, max_alpha)
        A = (log_omega, tau, min_logp, log_D, alpha)
        B = (log_omega, tau, max_logp, log_D, alpha)

        # Flatten A and B into a single row
        row = (*A, *B)
        couples.append(row)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "omega_A", "tau_A", "p_A", "D_A", "alpha_A",
            "omega_B", "tau_B", "p_B", "D_B", "alpha_B"
        ]
        writer.writerow(header)
        writer.writerows(couples)

    return filename

def load_trajectories_from_csv(filename):
    trajectories = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            # Convert each value to float and group into tuples of 5 parameters
            points = [
                list(map(float, row[i:i+5]))
                for i in range(0, len(row), 5)
            ]
            trajectories.append(points)

    return trajectories

import pandas as pd
from typing import List, Tuple

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

def generate_and_save_trajectories(seed, points_couples_filename: str, num_intermediate_samples: int, filename: str):

    random.seed(seed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    trajectories = []
    thetas_couples = load_and_extract_couples(points_couples_filename)
    for thetas_couple in thetas_couples:
        row = []
        row.extend(thetas_couple[0])

        intermediates = np.linspace(thetas_couple[0], thetas_couple[-1], num_intermediate_samples+2)[1:-1]
        for intermediate in intermediates:
            row.extend(intermediate)

        row.extend(thetas_couple[-1])

        trajectories.append(row)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filename
