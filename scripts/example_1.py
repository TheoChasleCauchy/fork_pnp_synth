import librosa
import numpy as np
import os

import torch
from pnp_synth.physical import ftm
import soundfile as sf

from fadtk import CLAPLaionModel
# Initialize the CLAP model
model = CLAPLaionModel(type="audio")
model.load_model()

nb_samples = 10

min_omega, max_omega = np.log10(40), np.log10(1000)
min_tau, max_tau = 0.4, 3
min_logp, max_logp = np.log10(10**-5), np.log10(0.2)
min_logD, max_logD = np.log10(10**-5), np.log10(0.3)
min_alpha, max_alpha = 10**-5, 1

center_omega = (max_omega + min_omega) / 2
center_tau = (max_tau + min_tau) / 2
center_logp = (max_logp + min_logp) / 2
center_logD = (max_logD + min_logD) / 2
center_alpha = (max_alpha + min_alpha) / 2

Theta_A = [center_omega, center_tau, min_logp, center_logD, center_alpha]
Theta_B = [center_omega, center_tau, max_logp, center_logD, center_alpha]
Theta_C = [max_omega, max_tau, center_logp, max_logD, max_alpha]
print(Theta_C)
assert 1==2

Theta_AC = [np.linspace(Theta_A, Theta_C, num=10)]
Theta_CB = [np.linspace(Theta_C, Theta_B, num=10)[1:]]

Theta_ACB = np.concatenate([Theta_AC, Theta_CB], axis=1)[0]
print(Theta_ACB)

def _load_audio(audio_path):
    """Helper to load audio file using librosa."""
    return librosa.load(audio_path, mono=True)[0]

# Create directory for audio files.
audio_dir = "generations/example_1/audios" # os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)
embeddings_dir = "generations/example_1/embeddings" # os.path.join(save_dir, "x")
os.makedirs(embeddings_dir, exist_ok=True)
logscale = True #the csv files are storing logscaled parameters

for i, row in enumerate(Theta_ACB):
    # Physical audio synthesis (g). theta -> x
    theta = np.array(row) # icassp23.THETA_COLUMNS
    x = ftm.rectangular_drum(theta, logscale, **ftm.constants).cpu()
    x = x / max(x)

    sf.write(os.path.join(audio_dir, f"audio_{i}_value_{row[2]:.2f}.wav"), x, ftm.constants["sr"])

for i, row in enumerate(Theta_ACB):
    audio = _load_audio(os.path.join("generations/example_1/audios", f"audio_{i}_value_{row[2]:.2f}.wav"))
    embedding = model._get_embedding(audio)
    audio_embedding = torch.mean(embedding, dim=0)
    audios_embeddings_np = np.array(audio_embedding.cpu().numpy())
    np.save(
        os.path.join("generations/example_1/embeddings", f"embedding_audio_{i}_value_{row[2]:.2f}"),
        audios_embeddings_np
    )