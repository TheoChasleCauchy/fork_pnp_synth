import argparse
import os  # Library for interacting with the operating system
from scipy.stats import pearsonr
import numpy as np
import re
import csv

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

def get_morphing_embeddings(embeddings_folder):

    npy_files = [f for f in os.listdir(embeddings_folder) if f.endswith(".npy")]

    parsed = []
    for f in npy_files:
        num_match = re.search(r'audio_(\d+)', f)
        val_match = re.search(r'value_(-?\d+\.?\d*)', f)
        if num_match and val_match:
            parsed.append((int(num_match.group(1)), float(val_match.group(1)), f))
    parsed.sort(key=lambda x: x[0])

    paths = [os.path.join(embeddings_folder, f) for _, _, f in parsed]
    values = [v for _, v, _ in parsed]

    # Load the embeddings from the numpy files
    embeddings = [np.load(path) for path in paths]

    return embeddings, values

def main(morphing_name):
    embeddings_folder = f"generations/{morphing_name}/embeddings"
    morphed_audios_embeddings, alpha_values = get_morphing_embeddings(embeddings_folder)

    pearson_cor, p = smoothness_clap_corr(morphed_audios_embeddings, alpha_values)

    # Write the values in a csv file
    with open(f"generations/{morphing_name}/{morphing_name}_smoothness_clap_corr.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pearson correlation", "P-value"])
        writer.writerow([pearson_cor, p])

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load the morphing's audios.")
    parser.add_argument("morphing_name", type=str, help="Name of the morphing")

    # Parse arguments
    args = parser.parse_args()

    # Call main with the directory argument
    main(args.morphing_name)