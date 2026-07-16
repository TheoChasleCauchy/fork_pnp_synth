import argparse
import csv
import re
import numpy as np  # Library for numerical operations
import os  # Library for interacting with the operating system
import librosa

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

def get_sources_and_middle_morphing_audio(audios_folder):
    audios = [os.path.join(audios_folder, f) for f in os.listdir(audios_folder) if f.endswith(".wav")]
    audios = sorted(audios, key=lambda x: int(re.search(r'audio_(\d+)', os.path.basename(x)).group(1)))
    return audios[0], audios[len(audios) // 2], audios[-1]

def main(morphing_name):
    audios_folder = f"generations/{morphing_name}/audios"

    source_audio, middle_morphing_audio, target_audio = get_sources_and_middle_morphing_audio(audios_folder)
    correspondence_value = correspondence_mfccs(source_audio, middle_morphing_audio, target_audio)
    
    # Write lcs values in a csv file
    with open(f"generations/{morphing_name}/{morphing_name}_correspondence_mfccs_value.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Correspondence MFCCs"])
        writer.writerow([correspondence_value])

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load the morphing's audios.")
    parser.add_argument("morphing_name", type=str, help="Name of the morphing")

    # Parse arguments
    args = parser.parse_args()

    # Call main with the directory argument
    main(args.morphing_name)