import csv, os
import numpy as np  # Library for numerical operations
from tqdm import tqdm

def correspondence_mfccs(mfcc_source_point: np.ndarray, mfcc_morphed_point: np.ndarray, mfcc_target_point: np.ndarray):
    """
    Load the morphed audio files and computes the MFCC coefficients.

    Args:
        mfcc_source_point (np.ndarray): The source audio point.
        mfcc_morphed_point (np.ndarray): The morphed audio point.
        mfcc_target_point (np.ndarray): The target audio point.

    Returns:
        metric (float): The computed metric value.
    """
    # Flatten the MFCC matrices to 1D arrays
    mfcc_source_point = np.array(mfcc_source_point)
    mfcc_morphed_point = np.array(mfcc_morphed_point)
    mfcc_target_point = np.array(mfcc_target_point)

    # Compute L2 norms
    norm_i_0 = np.linalg.norm(mfcc_morphed_point - mfcc_source_point)
    norm_i_last = np.linalg.norm(mfcc_morphed_point - mfcc_target_point)

    # Avoid division by zero
    denominator = norm_i_0 + norm_i_last
    assert denominator != 0.0

    ratio = norm_i_0 / denominator

    # Compute the coefficient
    coeff = np.abs(ratio - 0.5)

    return coeff.item()

def compute_soundmorpher_correspondence_mfccs(results_dir: str, model_name: str, trajectories: list[np.ndarray], embeddings_folder: str):

    correspondence_values = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing Correspondence MFCCs", total=len(trajectories))):
        morph_embeddings = []
        for i_theta in range(len(trajectory)):
            embedding = np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy"))
            morph_embeddings.append(embedding)

        source = morph_embeddings[0]
        middle = morph_embeddings[len(morph_embeddings)//2]
        target = morph_embeddings[-1]
        assert len(source) == len(middle) == len(target)

        correspondence_value = correspondence_mfccs(source, middle, target)
        correspondence_values.append(correspondence_value)

    # Write correspondence values in a csv file
    with open(f"{results_dir}/{model_name}/{model_name}_correspondence_mfccs_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Middle Morphing Audio Correspondence MFCCs"])
        for i, correspondence_value in enumerate(correspondence_values):
            writer.writerow([i, correspondence_value])
        writer.writerow(["Mean Correspondence", f"{np.mean(correspondence_values)} +- {np.std(correspondence_values)}"])
        
