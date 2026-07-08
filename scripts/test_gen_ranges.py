import librosa
import numpy as np
import os

import torch
from pnp_synth.physical import ftm
import soundfile as sf

range_omega = []
range_tau = []
range_logp = []
range_logD = []
range_alpha = []


def _load_audio(audio_path):
    """Helper to load audio file using librosa."""
    return librosa.load(audio_path, mono=True)[0]

# Create directory for audio files.
audio_dir = "generations/range_logp/audios" # os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)
logscale = True #the csv files are storing logscaled parameters

theta_rows = [[np.mean(range_omega), np.mean(range_tau), i, np.mean(range_logD), np.mean(range_alpha)] for i in range_logp]

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
    audio = _load_audio(os.path.join("generations/range_logp/audios", f"audio_{i}_value_logp_{row[2]:.2f}.wav"))
    embedding = model._get_embedding(audio)
    audio_embedding = torch.mean(embedding, dim=0)
    audios_embeddings_np = np.array(audio_embedding.cpu().numpy())
    np.save(
        os.path.join("generations/range_logp/embeddings", f"embedding_audio_{i}_value_logp_{row[2]:.2f}"),
        audios_embeddings_np
    )