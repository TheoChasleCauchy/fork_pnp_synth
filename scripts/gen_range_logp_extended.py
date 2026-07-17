import librosa
import numpy as np
import os

import torch
from pnp_synth.physical import ftm
import soundfile as sf

# def _load_audio(audio_path):
#     """Helper to load audio file using librosa."""
#     return librosa.load(audio_path, mono=True)[0]

# from fadtk import CLAPLaionModel
# # Initialize the CLAP model
# model = CLAPLaionModel(type="audio")
# model.load_model()

nb_samples = 19

range_omega = np.linspace(np.log10(40), np.log10(1000), num=nb_samples)
range_tau = np.linspace(0.4, 3, num=nb_samples)

range_logD = np.linspace(np.log10(10**-5), np.log10(0.3), num=nb_samples)
range_alpha = np.linspace(10**-5, 1, num=nb_samples)

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

range_logp = np.concatenate([np.linspace(np.log10(10**-5), -2.6, num=3, endpoint=False), np.linspace(-2.6, np.log10(0.2), num=22)])

# Create directory for audio files.
audio_dir = "generations/extended_range_logp/audios" # os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)
embeddings_dir = "generations/extended_range_logp/embeddings" # os.path.join(save_dir, "x")
os.makedirs(embeddings_dir, exist_ok=True)
logscale = True # the csv files are storing logscaled parameters

for i, logp_value in enumerate(range_logp):
    # Physical audio synthesis (g). theta -> x
    theta = np.array([2000, 0.4, 10**logp_value, 0.1, 0.5]) # icassp23.THETA_COLUMNS
    x = ftm.rectangular_drum(theta, False, **ftm.constants).cpu()
    x = x / max(x)

    sf.write(os.path.join(audio_dir, f"audio_{i}_value_{logp_value:.2f}.wav"), x, ftm.constants["sr"])

# for i, logp_value in enumerate(range_logp):
#     audio = _load_audio(os.path.join("generations/extended_range_logp/audios", f"audio_{i}_value_{logp_value:.2f}.wav"))
#     embedding = model._get_embedding(audio)
#     audio_embedding = torch.mean(embedding, dim=0)
#     audios_embeddings_np = np.array(audio_embedding.cpu().numpy())
#     np.save(
#         os.path.join("generations/extended_range_logp/embeddings", f"embedding_audio_{i}_value_{logp_value:.2f}"),
#         audios_embeddings_np
#     )