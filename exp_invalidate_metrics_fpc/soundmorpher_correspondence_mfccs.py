import csv
import re
import numpy as np  # Library for numerical operations
from tqdm import tqdm

def correspondence_mfccs(mfcc_source_point: tuple[float], mfcc_morphed_point: tuple[float], mfcc_target_point: tuple[float]):
    """
    Load the morphed audio files and computes the MFCC coefficients.

    Args:
        mfcc_source_point (tuple[float]): The source audio point.
        mfcc_morphed_point (tuple[float]): The morphed audio point.
        mfcc_target_point (tuple[float]): The target audio point.

    Returns:
        metric (float): The computed metric value.
    """
    # Step 2: Compute the coefficients for each tensor

    # Compute L2 norms
    norm_i_0 = np.linalg.norm(np.array(mfcc_morphed_point) - np.array(mfcc_source_point))
    norm_i_last = np.linalg.norm(np.array(mfcc_morphed_point) - np.array(mfcc_target_point))

    # Avoid division by zero
    denominator = norm_i_0 + norm_i_last
    if denominator == 0:
        ratio = 0.0
    else:
        ratio = norm_i_0 / denominator

    # Compute the coefficient
    coeff = abs(ratio - 0.5)

    return coeff

def compute_soundmorpher_correspondence_mfccs(points_csv: str, metric_csv: str, dimensions: int = 2):
    
    trajectories = []
    with open(points_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            # Convert each value to float and group into tuples of 5 parameters
            points = [
                list(map(float, row[i:i+dimensions]))
                for i in range(0, len(row), dimensions)
            ]
            trajectories.append(points)
    
    correspondence_values = []
    for trajectory in tqdm(trajectories, desc=f"Computing Correspondence on morphs", total=len(trajectories)):

        source = trajectory[0]
        middle = trajectory[len(trajectory)//2]
        target = trajectory[-1]

        correspondence_value = correspondence_mfccs(source, middle, target)
        correspondence_values.append(correspondence_value)

    # Write correspondence values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Middle Morphing Audio Correspondence MFCCs"])
        for i, correspondence_value in enumerate(correspondence_values):
            writer.writerow([i, correspondence_value])
        writer.writerow(["Mean Correspondence", f"{np.mean(correspondence_values)} +- {np.std(correspondence_values)}"])
        
