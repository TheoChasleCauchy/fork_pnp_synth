import argparse
import os  # Library for interacting with the operating system
from scipy.stats import pearsonr
import numpy as np
import re
import csv

import tqdm

def smoothness_clap_corr(morphed_audios_embeddings, alpha_values):
    """
    Compute the correlation between smoothness_values of CLAP embeddings of morphed audio files from their sources and the morphing parameter alpha.

    Args:
        morphed_audios_embeddings (list[np.ndarray]): List of morphed audio embeddings. The source is the first embedding of the list.
        alpha_values (list[float]): List of alpha values for interpolation.

    Returns:
        pearson_corr (float): The Pearson correlation coefficient.
        p (float): The p-value of the correlation.
    """
    assert len(morphed_audios_embeddings) == len(alpha_values), f"Length mismatch: {len(morphed_audios_embeddings)}, {len(alpha_values)}"
    
    smoothness_values = []
    for i in range(len(morphed_audios_embeddings)):
        dist = np.linalg.norm(morphed_audios_embeddings[i] - morphed_audios_embeddings[0]) # Assuming the source is the first embedding of the list
        smoothness_values.append(dist)

    pearson_corr, p = pearsonr(alpha_values, smoothness_values)

    return pearson_corr, p


def compute_smoothness_clap_corr():
    points_csv = f"exp_invalidate_metrics_example_1/generated/generated_intermediate_points.csv"
    
    trajectories = []
    with open(points_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            trajectories.append(row)
    
    pearson_corrs = []
    for trajectory in tqdm(trajectories, desc=f"Computing smoothness-CLAP correlation on morphs", total=len(trajectories)):
        # Stack the tuples in the list
        alpha_values = np.linspace(0, 1, len(trajectory))
        pearson_cor, p = smoothness_clap_corr(trajectory, alpha_values)
        pearson_corrs.append((pearson_cor, p))

    # Write the values in a csv file
    with open(f"exp_invalidate_metrics_example_1/generated/smoothness_clap_corr.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Pearson correlation", "P-value"])
        for i, (pearson_cor, p) in enumerate(pearson_corrs):
            writer.writerow([i, pearson_cor, p])
        writer.writerow(["Mean Pearson correlation", f"{np.mean([cor for cor, _ in pearson_corrs])} +- {np.std([cor for cor, _ in pearson_corrs])}"])
        