import csv
import torch
import numpy as np  # Library for numerical operations
from sklearn.decomposition import PCA
import csv
from tqdm import tqdm

def compute_lcs(middle_point):
    """
    Compute the Latent Component Score (LCS) for a list of morphed audio clips.

    Args:
        middle_point: matrix of shape (n_samples, n_features).

    Returns:
        LCS value.
    """

    with torch.no_grad():
        # Step 1: Extract latent features using DAC
        latents = middle_point

        # Step 2: Apply PCA to the latent features
        pca = PCA(n_components=2)
        pca.fit(latents)  # Reshape to 2D for PCA

        # Step 3: Compute cumulative variance explained by the first two components
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.sum(explained_variance[:2])  # PC1 + PC2

        # Step 4: LCS is the cumulative variance
        lcs_value = cumulative_variance

    return lcs_value

def compute_mix2morph_lcs(points_csv: str, metric_csv: str, dimensions=2):
    
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
    
    lcs_values = []
    for trajectory in tqdm(trajectories, desc=f"Computing LCS on morphs", total=len(trajectories)):
        # Stack the tuples in the list
        trajectory = np.array(trajectory)

        lcs_value = compute_lcs(trajectory)
        lcs_values.append(lcs_value)

    # Write lcs values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Middle Morphing Audio LCS"])
        for i, lcs_value in enumerate(lcs_values):
            writer.writerow([i, lcs_value])
        writer.writerow(["Mean LCS", f"{np.mean(lcs_values)} +- {np.std(lcs_values)}"])
        
