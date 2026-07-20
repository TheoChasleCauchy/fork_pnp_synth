import librosa
import numpy as np
import os

import torch
from pnp_synth.physical import ftm
import soundfile as sf

def _load_audio(audio_path):
    """Helper to load audio file using librosa."""
    return librosa.load(audio_path, mono=True)[0]

# Create directory for audio files.
audio_dir = "generations/metal_to_wood/audios" # os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)
embeddings_dir = "generations/metal_to_wood/embeddings" # os.path.join(save_dir, "x")
os.makedirs(embeddings_dir, exist_ok=True)
logscale = True #the csv files are storing logscaled parameters

# Hypercube: [[1500, 8000], [0.015, 1.0], [0.15, 1000], [-5, -0.5], [0.01, 1.0]]
theta_rows = [[3.102961, 0.2, i, -2.723314, 0.500034] for i in np.linspace(-1.1, 0.5, num=10)]

for i, row in enumerate(theta_rows):
    # Physical audio synthesis (g). theta -> x
    theta = np.array(row) # icassp23.THETA_COLUMNS
    x = ftm.rectangular_drum(theta, logscale, **ftm.constants).cpu()
    x = x / max(x)

    sf.write(os.path.join(audio_dir, f"audio_{i}_value_{row[2]:.2f}.wav"), x, ftm.constants["sr"])


from fadtk import CLAPLaionModel
# Initialize the CLAP model
model = CLAPLaionModel(type="audio")
model.load_model()

for i, row in enumerate(theta_rows):
    audio = _load_audio(os.path.join("generations/metal_to_wood/audios", f"audio_{i}_value_{row[2]:.2f}.wav"))
    embedding = model._get_embedding(audio)
    audio_embedding = torch.mean(embedding, dim=0)
    audios_embeddings_np = np.array(audio_embedding.cpu().numpy())
    np.save(
        os.path.join("generations/metal_to_wood/embeddings", f"embedding_audio_{i}_value_{row[2]:.2f}"),
        audios_embeddings_np
    )
