import argparse
import csv
import re
from dac import model
import numpy as np  # Library for numerical operations
import os  # Library for interacting with the operating system
import librosa
import tqdm

def correspondence_mfccs(source_audio_name: str, morphed_audios_name: str, target_audio_name: str):
    """
    Load the morphed audio files and computes the MFCC coefficients.

    Args:
        source_audio_name (str): Path to the source audio file.
        morphed_audios_name (str): Path to the morphed audio file.
        target_audio_name (str): Path to the target audio file.

    Returns:
        metric (float): The computed metric value.
    """
    source_audio = librosa.load(source_audio_name)[0]
    morphed_audio = librosa.load(morphed_audios_name)[0]
    target_audio = librosa.load(target_audio_name)[0]

    # Step 1: Compute MFCCs for each tensor
    mfcc_source = librosa.feature.mfcc(y=source_audio)
    mfcc_morphed = librosa.feature.mfcc(y=morphed_audio)
    mfcc_target = librosa.feature.mfcc(y=target_audio)

    # Flatten the MFCC matrices to 1D arrays
    mfcc_flat_source = mfcc_source.flatten()
    mfcc_flat_morphed = mfcc_morphed.flatten()
    mfcc_flat_target = mfcc_target.flatten()

    # Step 2: Compute the coefficients for each tensor

    # Compute L2 norms
    norm_i_0 = np.linalg.norm(mfcc_flat_morphed - mfcc_flat_source)
    norm_i_last = np.linalg.norm(mfcc_flat_morphed - mfcc_flat_target)

    # Avoid division by zero
    denominator = norm_i_0 + norm_i_last
    if denominator == 0:
        ratio = 0.0
    else:
        ratio = norm_i_0 / denominator

    # Compute the coefficient
    coeff = abs(ratio - 0.5)

    return coeff

def compute_soundmorpher_correspondence_mfccs(thetas_couples, num_intermediate_samples):
    audios_folder = f"exp_embeddings_linearity/generated/audio"

    def _load_audio(audio_path):
        """Helper to load audio file using librosa."""
        return librosa.load(audio_path, mono=True)[0]
    
    correspondence_values = []
    for i in tqdm(thetas_couples, desc=f"Computing Correspondence MFCCs on morphs", total=len(thetas_couples)):

        # Synthesize intermediate samples
        alpha = num_intermediate_samples//2

        source = _load_audio(os.path.join(audios_folder, f"audio_row_{i}_A.wav"))
        middle = _load_audio(os.path.join(audios_folder, f"audio_row_{i}_AB_I{alpha}.wav"))
        target = _load_audio(os.path.join(audios_folder, f"audio_row_{i}_B.wav"))

        correspondence_value = correspondence_mfccs(source, middle, target)
        correspondence_values.append(correspondence_value)

    # Write correspondence values in a csv file
    with open(f"exp_embeddings_linearity/generated/soundmorpher_correspondence_mfccs_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Middle Morphing Audio Correspondence MFCCs"])
        for value in correspondence_values:
            writer.writerow([value])
