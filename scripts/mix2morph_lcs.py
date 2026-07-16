import argparse
import torch
import numpy as np  # Library for numerical operations
import os  # Library for interacting with the operating system
import re
from sklearn.decomposition import PCA
import dac
from audiotools import AudioSignal
import csv

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.sample_rate = 24000
model.to('cuda')

def extract_dac_latents(audio):
    """
    Extract latent features for an audio clip using the Descript Audio Codec (DAC) model.

    Args:
        audio: Audio path or signal.
        sample_rate: Sample rate of the audio (default: 24000, as expected by DAC).

    Returns:
        Latent features (numpy array).
    """
    signal = AudioSignal(audio, sample_rate=model.sample_rate)
    signal.to(model.device)
    x = model.preprocess(signal.audio_data, model.sample_rate)
    _, _, latents, _, _ = model.encode(x)

    return latents

def compute_lcs(morphed_audios):
    """
    Compute the Latent Component Score (LCS) for a list of morphed audio clips.

    Args:
        morphed_audios: List of audio signals (1D numpy arrays).

    Returns:
        List containing the mean LCS and individual LCS values.
    """
    lcs_values = []

    for audio in morphed_audios:
        with torch.no_grad():
            # Step 1: Extract latent features using DAC
            latents = extract_dac_latents(audio).cpu()
            latents = latents[0,:,:]
            print(latents.shape)

            # Step 2: Apply PCA to the latent features
            pca = PCA(n_components=2)
            pca.fit(latents)  # Reshape to 2D for PCA

            # Step 3: Compute cumulative variance explained by the first two components
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.sum(explained_variance[:2])  # PC1 + PC2

            # Step 4: LCS is the cumulative variance
            lcs = cumulative_variance
            lcs_values.append(lcs)

    # Compute mean LCS
    mean_lcs = np.mean(lcs_values)

    # Return [mean_LCS, LCS1, LCS2, ...]
    return [mean_lcs] + lcs_values

def get_middle_morphing_audio(audios_folder):
    audios = [os.path.join(audios_folder, f) for f in os.listdir(audios_folder) if f.endswith(".wav")]
    audios = sorted(audios, key=lambda x: int(re.search(r'audio_(\d+)', os.path.basename(x)).group(1)))
    return audios[len(audios) // 2]

def main(morphing_name):
    audios_folder = f"generations/{morphing_name}/audios"

    morphing_audio = get_middle_morphing_audio(audios_folder)
    lcs_value = compute_lcs([morphing_audio])
    
    # Write lcs values in a csv file
    with open(f"generations/{morphing_name}/{morphing_name}_lcs_value.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Middle Morphing Audio LCS"])
        writer.writerow([lcs_value])

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load the morphing's audios.")
    parser.add_argument("morphing_name", type=str, help="Name of the morphing")

    # Parse arguments
    args = parser.parse_args()

    # Call main with the directory argument
    main(args.morphing_name)