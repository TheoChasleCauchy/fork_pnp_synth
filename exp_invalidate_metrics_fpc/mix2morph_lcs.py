import csv
import torch
import numpy as np  # Library for numerical operations
from sklearn.decomposition import PCA
import csv
from tqdm import tqdm

from exp_functions import embeddings

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

def compute_mix2morph_lcs(dirname: str, metric_csv: str, morph_type: str):
    # Load trajectories tensor
    trajectories_tensor_path = f"{dirname}/dac_{morph_type}_trajectories.pt"
    trajectories_tensor = torch.load(trajectories_tensor_path)

    lcs_values = []
    for trajectory in tqdm(trajectories_tensor, desc=f"Computing LCS on morphs", total=len(trajectories_tensor)):
        middle_point = trajectory[len(trajectory) // 2]
        assert middle_point.shape == torch.Size(embeddings['dac']), f"Expected shape {embeddings['dac']}, got {middle_point.shape}"

        lcs_value = compute_lcs(middle_point)
        lcs_values.append(lcs_value)

    # Write lcs values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Middle Morphing Audio LCS"])
        for i, lcs_value in enumerate(lcs_values):
            writer.writerow([i, lcs_value])
        writer.writerow(["Mean LCS", f"{np.mean(lcs_values)} +- {np.std(lcs_values)}"])
        
