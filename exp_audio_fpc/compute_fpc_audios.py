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
min_log_p, max_log_p = np.log10(min_p), np.log10(max_p)
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
                distance_from_a = np.linalg.norm(a - b) * i / (num_intermediate_samples + 1)
                intermediate_point = a + distance_from_a * random_direction
                if (min_log_omega <= intermediate_point[0] <= max_log_omega and
                    min_tau <= intermediate_point[1] <= max_tau and
                    min_log_p <= intermediate_point[2] <= max_log_p and
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

def create_normalized_fpcc_intermediate_points(num_intermediate_samples, dirname):

    def normalize_theta(theta):
        # Normalize between 0 and 1 each parameter
        assert len(theta) == 5
        d_log_omega = max_log_omega - min_log_omega
        normalized_log_omega = (theta[0] - min_log_omega) / d_log_omega

        d_tau = max_tau - min_tau
        normalized_tau = (theta[1] - min_tau) / d_tau

        d_log_p = max_log_p - min_log_p
        normalized_log_p = (theta[2] - min_log_p) / d_log_p

        d_log_D = max_log_D - min_log_D
        normalized_log_D = (theta[3] - min_log_D) / d_log_D

        d_alpha = max_alpha - min_alpha
        normalized_alpha = (theta[4] - min_alpha) / d_alpha

        return np.array([normalized_log_omega, normalized_tau, normalized_log_p, normalized_log_D, normalized_alpha])

    def denormalize_theta(theta):
        # Denormalize from between 0 and 1 each parameter to their range
        assert len(theta) == 5
        d_log_omega = max_log_omega - min_log_omega
        denormalized_log_omega = theta[0] * d_log_omega + min_log_omega

        d_tau = max_tau - min_tau
        denormalized_tau = theta[1] * d_tau + min_tau

        d_log_p = max_log_p - min_log_p
        denormalized_log_p = theta[2] * d_log_p + min_log_p

        d_log_D = max_log_D - min_log_D
        denormalized_log_D = theta[3] * d_log_D + min_log_D

        d_alpha = max_alpha - min_alpha
        denormalized_alpha = theta[4] * d_alpha + min_alpha

        return np.array([denormalized_log_omega, denormalized_tau, denormalized_log_p, denormalized_log_D, denormalized_alpha])

    # Load couples as torch tensor
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    trajectories = []
    for couple in tqdm(couples, desc=f"Computing normalized FPCC trajectories"):
        a, b = np.array(couple[0]), np.array(couple[1])
        # Normalize a
        normalized_a = normalize_theta(a)
        normalized_b = normalize_theta(b)

        # Initialize trajectory with point A
        trajectory = []
        trajectory.extend(normalized_a)

        # Generate intermediate points between A and B
        for i in range(1, num_intermediate_samples + 1):
            valid_parameters = False
            while not valid_parameters:

                # Random direction in the same shape as A
                random_direction = np.random.rand(len(a))
                random_direction = random_direction / np.linalg.norm(random_direction)  # Normalize

                # Distance from A proportional to progress
                distance_from_a = np.linalg.norm(normalized_a - normalized_b) * i / (num_intermediate_samples + 1)
                intermediate_point = normalized_a + distance_from_a * random_direction
                if (0.0 <= intermediate_point[0] <= 1.0 and
                    0.0 <= intermediate_point[1] <= 1.0 and
                    0.0 <= intermediate_point[2] <= 1.0 and
                    0.0 <= intermediate_point[3] <= 1.0 and
                    0.0 <= intermediate_point[4] <= 1.0):
                    valid_parameters = True
                denormalized_intermediate_point = denormalize_theta(intermediate_point)

            trajectory.extend(denormalized_intermediate_point)

        trajectory.extend(normalized_b)
        assert len(trajectory) == (num_intermediate_samples + 2)*5, f"Expected {(num_intermediate_samples + 2)*5} points, got {len(trajectory)}"

        trajectories.append(trajectory)

    # Save to CSV
    os.makedirs(dirname, exist_ok=True)
    filepath = os.path.join(dirname, "normalized_fpcc_trajectories.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filepath

def get_embeddings_fpcc_intermediate_points(embedding_model, num_intermediate_samples, random_points_embeddings_dir, trajectories_embeddings_dir):

    import shutil

    random_points_embeddings_dir = os.path.join(random_points_embeddings_dir, embedding_model)
    trajectories_embeddings_dir = os.path.join(trajectories_embeddings_dir, embedding_model)
    os.makedirs(trajectories_embeddings_dir, exist_ok=True)

    # def normalize_embedings(theta):
    #     pass

    def load_random_points_from_csv(embeddings_dir):
        random_points_embeddings = []

        for root, _, files in os.walk(embeddings_dir):
            for filename in tqdm(files, desc=f"Loading random points from {embeddings_dir}"):
                if filename.endswith(".npy"):
                    filepath = os.path.join(root, filename)
                    random_points_embeddings.append({"filepath": filepath, "embedding": np.load(filepath)})

        return random_points_embeddings

    def find_closest_point_from_circle(source_point, distance_from_source, points):
        closest_point_filepath = None
        min_distance = float('inf')

        for point in points:
            dist = np.linalg.norm(np.array(point["embedding"]) - np.array(source_point))
            temp_dist = abs(dist - distance_from_source)
            if temp_dist < min_distance:
                min_distance = temp_dist
                closest_point_filepath = point["filepath"]

        return closest_point_filepath

    # Load couples as torch tensor
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    random_points = load_random_points_from_csv(random_points_embeddings_dir)

    for i_couple in tqdm(range(len(couples)), desc=f"Computing embeddings FPCC trajectories"):
        a_filepath = os.path.join(random_points_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I0.npy")
        shutil.copy(a_filepath, trajectories_embeddings_dir)
        b_filepath = os.path.join(random_points_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I{num_intermediate_samples+1}.npy")
        shutil.copy(b_filepath, trajectories_embeddings_dir)

        # Load embeddings for points A and B
        a = np.load(a_filepath)
        b = np.load(b_filepath)

        # Normalize a and b
        # normalized_a = normalize_theta(a)
        # normalized_b = normalize_theta(b)

        # Generate intermediate points between A and B
        for i in range(1, num_intermediate_samples + 1):
            # Distance from A proportional to progress
            distance_from_a = np.linalg.norm(b - a) * i / (num_intermediate_samples + 1)

            intermediate_point_filepath = find_closest_point_from_circle(a, distance_from_a, random_points)
            shutil.copyfile(intermediate_point_filepath, os.path.join(trajectories_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I{i}.npy"))


    return trajectories_embeddings_dir