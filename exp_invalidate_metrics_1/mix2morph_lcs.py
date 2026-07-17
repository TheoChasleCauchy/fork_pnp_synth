import librosa
import torch
import numpy as np  # Library for numerical operations
import os  # Library for interacting with the operating system
import re
from sklearn.decomposition import PCA
import dac
from audiotools import AudioSignal
import csv
import tqdm

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

def compute_lcs(morphed_audio):
    """
    Compute the Latent Component Score (LCS) for a list of morphed audio clips.

    Args:
        morphed_audio: Audio signal (1D numpy array).

    Returns:
        LCS value.
    """

    with torch.no_grad():
        # Step 1: Extract latent features using DAC
        latents = extract_dac_latents(morphed_audio).cpu()
        latents = latents[0,:,:]
        print(latents.shape)

        # Step 2: Apply PCA to the latent features
        pca = PCA(n_components=2)
        pca.fit(latents)  # Reshape to 2D for PCA

        # Step 3: Compute cumulative variance explained by the first two components
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.sum(explained_variance[:2])  # PC1 + PC2

        # Step 4: LCS is the cumulative variance
        lcs_value = cumulative_variance

    return lcs_value

def compute_mix2morph_lcs(thetas_couples, num_intermediate_samples):
    audios_folder = f"exp_embeddings_linearity/generated/audio"

    def _load_audio(audio_path):
        """Helper to load audio file using librosa."""
        return librosa.load(audio_path, mono=True)[0]
    
    lcs_values = []
    for i in tqdm(thetas_couples, desc=f"Computing LCS on morphs", total=len(thetas_couples)):

        # Synthesize intermediate samples
        alpha = num_intermediate_samples//2

        audio = _load_audio(os.path.join(audios_folder, f"audio_row_{i}_AB_I{alpha}.wav"))

        lcs_value = compute_lcs([audio])
        lcs_values.append(lcs_value)

    # Write lcs values in a csv file
    with open(f"exp_embeddings_linearity/generated/mix2morph_lcs_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Middle Morphing Audio LCS"])
        for lcs_value in lcs_values:
            writer.writerow([lcs_value])
