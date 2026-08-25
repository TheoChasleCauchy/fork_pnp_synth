from tqdm import tqdm
import csv
import pandas as pd
from typing import List, Tuple
import numpy as np
import os
import random

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


def create_nuc_intermediate_points(num_intermediate_samples, dirname):
    alpha_values = [0.01 * i for i in range(1, num_intermediate_samples)] + [0.99]

    # Load couples as torch tensor
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    trajectories = []
    for couple in tqdm(couples, desc=f"Computing NUC trajectories"):
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
    filepath = os.path.join(dirname, "nuc_trajectories.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filepath


def create_eqc_intermediate_points(num_intermediate_samples, dirname):

    # Load couples as torch tensor
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    trajectories = []
    for couple in tqdm(couples, desc=f"Computing EQC trajectories"):
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
    filepath = os.path.join(dirname, "eqc_trajectories.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filepath

def create_normalized_eqc_intermediate_points(num_intermediate_samples, dirname):

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
    for couple in tqdm(couples, desc=f"Computing normalized EQC trajectories"):
        a, b = np.array(couple[0]), np.array(couple[1])
        # Normalize a
        normalized_a = normalize_theta(a)
        normalized_b = normalize_theta(b)

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

        trajectory.extend(b)
        assert len(trajectory) == (num_intermediate_samples + 2)*5, f"Expected {(num_intermediate_samples + 2)*5} points, got {len(trajectory)}"

        trajectories.append(trajectory)

    # Save to CSV
    os.makedirs(dirname, exist_ok=True)
    filepath = os.path.join(dirname, "normalized_eqc_trajectories.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filepath

def get_embeddings_eqc_intermediate_points(embedding_model, num_intermediate_samples, random_points_embeddings_dir, trajectories_embeddings_dir, results_dir):
    """
    Generate intermediate embedding points along circular trajectories between endpoint couples (A, B).
    For each couple, finds points at specific distances from A that approximate a circular arc to B,
    saves these to the trajectories directory, and records the angular deviation from the direct AB line.

    Args:
        embedding_model: Name identifier for the embedding model
        num_intermediate_samples: Number of intermediate points between each A-B couple
        random_points_embeddings_dir: Base directory containing random embedding points
        trajectories_embeddings_dir: Base directory to save generated trajectory embeddings
        results_dir: Base directory to save results

    Returns:
        Path to the directory containing all generated trajectory embeddings and angle data
    """

    import shutil
    import csv

    # Set up model-specific subdirectories
    random_points_embeddings_dir = os.path.join(random_points_embeddings_dir, embedding_model)
    trajectories_embeddings_dir = os.path.join(trajectories_embeddings_dir, embedding_model)
    results_dir = os.path.join(results_dir, embedding_model)
    os.makedirs(trajectories_embeddings_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    def load_random_points_from_csv(embeddings_dir):
        """Recursively load all embedding .npy files from directory tree."""
        random_points_embeddings = []

        for root, _, files in os.walk(embeddings_dir):
            for filename in tqdm(files, desc=f"Loading random points from {embeddings_dir}"):
                if filename.endswith(".npy"):
                    filepath = os.path.join(root, filename)
                    random_points_embeddings.append({"filepath": filepath, "embedding": np.load(filepath)})

        return random_points_embeddings

    def find_closest_point_from_circle(source_point, distance_from_source, points):
        """
        Find the point whose distance from source_point is closest to the specified distance.
        This implements a circular search rather than linear interpolation.
        """
        closest_point_filepath = None
        min_distance = float('inf')

        for point in points:
            dist = np.linalg.norm(np.array(point["embedding"]) - np.array(source_point))
            temp_dist = abs(dist - distance_from_source)
            if temp_dist < min_distance:
                min_distance = temp_dist
                closest_point_filepath = point["filepath"]

        return closest_point_filepath

    def calculate_angle(a, b, intermediate):
        """
        Calculate the angle in degrees between the AB line and the line from A to the intermediate point.
        This measures how much the closest found point deviates from the direct path from A to B.
        """
        vec_ai = intermediate - a
        vec_ab = b - a

        norm_ai = np.linalg.norm(vec_ai)
        norm_ab = np.linalg.norm(vec_ab)

        if norm_ai == 0 or norm_ab == 0:
            return 0.0

        cos_angle = np.dot(vec_ai, vec_ab) / (norm_ai * norm_ab)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # Load couples (A, B pairs) from CSV
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    # Load all available random points for nearest-neighbor search
    random_points = load_random_points_from_csv(random_points_embeddings_dir)

    # Storage for angle measurements
    angles_data = []

    for i_couple in tqdm(range(len(couples)), desc=f"Computing embeddings EQC trajectories"):
        # Copy endpoint A (I0) and B (I{num_intermediate_samples+1}) to trajectories directory
        a_filepath = os.path.join(random_points_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I0.npy")
        shutil.copy(a_filepath, trajectories_embeddings_dir)
        b_filepath = os.path.join(random_points_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I{num_intermediate_samples+1}.npy")
        shutil.copy(b_filepath, trajectories_embeddings_dir)

        # Load endpoint embeddings
        a = np.load(a_filepath)
        b = np.load(b_filepath)

        # Generate intermediate points between A and B using circular search
        for i in range(1, num_intermediate_samples + 1):
            # Target distance from A for this intermediate point
            distance_from_a = np.linalg.norm(b - a) * i / (num_intermediate_samples + 1)

            # Find point whose distance from A is closest to the target distance
            intermediate_point_filepath = find_closest_point_from_circle(a, distance_from_a, random_points)
            shutil.copyfile(
                intermediate_point_filepath,
                os.path.join(trajectories_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I{i}.npy")
            )

            # Load the actual intermediate embedding to calculate deviation angle
            intermediate_embedding = np.load(intermediate_point_filepath)
            angle_deg = calculate_angle(a, b, intermediate_embedding)
            angles_data.append({
                'row_index': i_couple,
                'intermediate_index': i,
                'angle_degrees': angle_deg
            })

    # Save all angle measurements to CSV in the trajectories directory
    angles_filepath = os.path.join(results_dir, 'intermediate_points_angles.csv')
    with open(angles_filepath, 'w', newline='') as csvfile:
        fieldnames = ['row_index', 'intermediate_index', 'angle_degrees']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(angles_data)

    return trajectories_embeddings_dir

def get_embeddings_nuc_intermediate_points(embedding_model, num_intermediate_samples, random_points_embeddings_dir, trajectories_embeddings_dir, results_dir):
    """
    Generate intermediate embedding points along linear trajectories between endpoint couples (A, B).
    For each couple, interpolates points along the line from A to B, finds the closest actual random embedding,
    saves these to the trajectories directory, and records the angular deviation from the ideal line.

    Args:
        embedding_model: Name identifier for the embedding model
        num_intermediate_samples: Number of intermediate points between each A-B couple
        random_points_embeddings_dir: Base directory containing random embedding points
        trajectories_embeddings_dir: Base directory to save generated trajectory embeddings
        results_dir: Base directory to save results

    Returns:
        Path to the directory containing all generated trajectory embeddings and angle data
    """

    import shutil
    import csv

    # Set up model-specific subdirectories
    random_points_embeddings_dir = os.path.join(random_points_embeddings_dir, embedding_model)
    trajectories_embeddings_dir = os.path.join(trajectories_embeddings_dir, embedding_model)
    results_dir = os.path.join(results_dir, embedding_model)
    os.makedirs(trajectories_embeddings_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    def load_random_points_from_csv(embeddings_dir):
        """Recursively load all embedding .npy files from directory tree."""
        random_points_embeddings = []

        for root, _, files in os.walk(embeddings_dir):
            for filename in tqdm(files, desc=f"Loading random points from {embeddings_dir}"):
                if filename.endswith(".npy"):
                    filepath = os.path.join(root, filename)
                    random_points_embeddings.append({"filepath": filepath, "embedding": np.load(filepath)})

        return random_points_embeddings

    def find_closest_point(reference_point, points):
        """Find the embedding point with minimum Euclidean distance to the reference point."""
        closest_point_filepath = None
        min_distance = float('inf')

        for point in points:
            temp_dist = np.linalg.norm(np.array(point["embedding"]) - np.array(reference_point))
            if temp_dist < min_distance:
                min_distance = temp_dist
                closest_point_filepath = point["filepath"]

        return closest_point_filepath

    def calculate_angle(a, b, intermediate):
        """
        Calculate the angle in degrees between the AB line and the line from A to the intermediate point.
        This measures how much the closest found point deviates from the ideal linear interpolation.
        """
        vec_ai = intermediate - a
        vec_ab = b - a

        norm_ai = np.linalg.norm(vec_ai)
        norm_ab = np.linalg.norm(vec_ab)

        if norm_ai == 0 or norm_ab == 0:
            return 0.0

        cos_angle = np.dot(vec_ai, vec_ab) / (norm_ai * norm_ab)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # Alpha values for linear interpolation: 0.01, 0.02, ..., 0.99
    alpha_values = [0.01 * i for i in range(1, num_intermediate_samples)] + [0.99]

    # Load the couples (A, B pairs) from CSV
    couples = load_and_extract_couples(f"exp_embeddings_linearity/generated/thetas_couples.csv")

    # Load all available random points for nearest-neighbor search
    random_points = load_random_points_from_csv(random_points_embeddings_dir)

    # Storage for angle measurements
    angles_data = []

    for i_couple in tqdm(range(len(couples)), desc=f"Computing embeddings NUC trajectories"):
        # Copy endpoint A (I0) and B (I{num_intermediate_samples+1}) to trajectories directory
        a_filepath = os.path.join(random_points_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I0.npy")
        shutil.copy(a_filepath, trajectories_embeddings_dir)
        b_filepath = os.path.join(random_points_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I{num_intermediate_samples+1}.npy")
        shutil.copy(b_filepath, trajectories_embeddings_dir)

        # Load endpoint embeddings
        a = np.load(a_filepath)
        b = np.load(b_filepath)

        for i in range(1, num_intermediate_samples + 1):
            # Ideal interpolated point at fraction alpha_values[i-1] along AB
            vector_from_a = (b - a) * alpha_values[i - 1]
            ideal_point = a + vector_from_a

            # Find the actual random point closest to this ideal point
            intermediate_point_filepath = find_closest_point(ideal_point, random_points)
            shutil.copyfile(
                intermediate_point_filepath,
                os.path.join(trajectories_embeddings_dir, f"embedding_{embedding_model}_row_{i_couple}_AB_I{i}.npy")
            )

            # Load the actual intermediate embedding to calculate deviation angle
            intermediate_embedding = np.load(intermediate_point_filepath)
            angle_deg = calculate_angle(a, b, intermediate_embedding)
            angles_data.append({
                'row_index': i_couple,
                'intermediate_index': i,
                'angle_degrees': angle_deg
            })

    # Save all angle measurements to CSV in the trajectories directory
    angles_filepath = os.path.join(results_dir, 'intermediate_points_angles.csv')
    with open(angles_filepath, 'w', newline='') as csvfile:
        fieldnames = ['row_index', 'intermediate_index', 'angle_degrees']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(angles_data)

    return trajectories_embeddings_dir

def generate_random_points(num_points, points_filename, seed):
    random.seed(seed)
    os.makedirs(os.path.dirname(points_filename), exist_ok=True)

    points = []
    for _ in range(num_points):
        theta = []
        # Generate random parameters for each point
        theta.append(random.uniform(min_log_omega, max_log_omega))
        theta.append(random.uniform(min_tau, max_tau))
        theta.append(random.uniform(min_log_p, max_log_p))
        theta.append(random.uniform(min_log_D, max_log_D))
        theta.append(random.uniform(min_alpha, max_alpha))

        points.append(theta)

    # Save to CSV
    with open(points_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}" for param in ["log_omega", "tau", "log_p", "log_D", "alpha"]]
        writer.writerow(header)
        for theta in points:
            writer.writerow(theta)

    return points_filename
