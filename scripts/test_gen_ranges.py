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

nb_samples = 19

range_omega = np.linspace(np.log10(40), np.log10(1000), num=nb_samples)
# range_omega = np.log10(range_omega)
range_tau = np.linspace(0.4, 3, num=nb_samples)
range_logp = np.linspace(np.log10(10**-5), np.log10(0.2), num=nb_samples)
# range_logp = np.log10(range_p)
range_logD = np.linspace(np.log10(10**-5), np.log10(0.3), num=nb_samples)
# range_logD = np.log10(range_D)
range_alpha = np.linspace(10**-5, 1, num=nb_samples)

average_omega = np.mean(range_omega)
average_tau = np.mean(range_tau)
average_logp = np.mean(range_logp)
average_logD = np.mean(range_logD)
average_alpha = np.mean(range_alpha)

parameters = {
    "omega": {
        "range": range_omega,
        "average": average_omega
    },
    "tau": {
        "range": range_tau,
        "average": average_tau
    },
    "logp": {
        "range": range_logp,
        "average": average_logp
    },
    "logD": {
        "range": range_logD,
        "average": average_logD
    },
    "alpha": {
        "range": range_alpha,
        "average": average_alpha
    }
}


def _load_audio(audio_path):
    """Helper to load audio file using librosa."""
    return librosa.load(audio_path, mono=True)[0]


for i_parameter, parameter in enumerate(parameters.keys()):

    # Create directory for audio files.
    audio_dir = f"generations/range_{parameter}/audios"
    os.makedirs(audio_dir, exist_ok=True)
    logscale = True # logscaled parameters

    theta_rows = []
    # [i, a, a ,a ,a], [a, i, a ,a ,a], ...
    for i in range(nb_samples):
        row = []
        for param in parameters.keys():
            if param == parameter:
                element = parameters[param]["range"][i]
            else:
                element = parameters[param]["average"]
            row.append(element)
        theta_rows.append(row)

    for i, row in enumerate(theta_rows):
        # Physical audio synthesis (g). theta -> x
        theta = np.array(row) # icassp23.THETA_COLUMNS
        x = ftm.rectangular_drum(theta, logscale, **ftm.constants).cpu()
        x = x / max(x)

        sf.write(os.path.join(audio_dir, f"audio_{i}_param_{parameter}_value_{row[i_parameter]:.2f}.wav"), x, ftm.constants["sr"])

    embeddings_dir = f"generations/range_{parameter}/embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)

    for i, row in enumerate(theta_rows):
        audio = _load_audio(os.path.join(audio_dir, f"audio_{i}_param_{parameter}_value_{row[i_parameter]:.2f}.wav"))
        embedding = model._get_embedding(audio)
        audio_embedding = torch.mean(embedding, dim=0)
        audios_embeddings_np = np.array(audio_embedding.cpu().numpy())
        np.save(
            os.path.join(embeddings_dir, f"embedding_audio_{i}_param_{parameter}_value_{row[i_parameter]:.2f}"),
            audios_embeddings_np
        )