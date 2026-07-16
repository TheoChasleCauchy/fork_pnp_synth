import csv
import os
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from pnp_synth.physical import ftm

from load_models_and_audios import _load_audio, _load_model
from generate_thetas import generate_and_save_couples

seed = 42

# 1. Generate couples of points in a 2D space
points_couples_filename = generate_and_save_couples(seed, 1000)

# 2. Load the generated couples
thetas_couples = []
with open(points_couples_filename, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        # Convert each value to float and create tuples for A and B
        values = [float(x) for x in row]
        A = tuple(values[:5])
        B = tuple(values[5:10])
        thetas_couples.append((A, B))

# 3. Generate the audios
audio_dir = "exp_embeddings_linearity/generated/audio"
os.makedirs(audio_dir, exist_ok=True)

num_intermediate_samples = 11

for i, couple in enumerate(tqdm(thetas_couples, desc="Processing couples", total=len(thetas_couples))):

    # Synthesize sources
    Theta_A, Theta_B = couple

    # Physical audio synthesis (g). theta -> x
    x = ftm.rectangular_drum(Theta_A, True, **ftm.constants).cpu()
    x = x / max(x)

    sf.write(os.path.join(audio_dir, f"audio_row_{i}_A.wav"), x, ftm.constants["sr"])

    # Physical audio synthesis (g). theta -> x
    x = ftm.rectangular_drum(Theta_B, True, **ftm.constants).cpu()
    x = x / max(x)

    sf.write(os.path.join(audio_dir, f"audio_row_{i}_B.wav"), x, ftm.constants["sr"])

    # Synthesize intermediate samples
    range_AB = np.linspace(Theta_A, Theta_B, num=num_intermediate_samples, endpoint=False)[1:] # Do not resynthesize A

    for alpha, theta in enumerate(range_AB):
        x = ftm.rectangular_drum(theta, True, **ftm.constants).cpu()
        x = x / max(x)
        sf.write(os.path.join(audio_dir, f"audio_row_{i}_AB_I{alpha}.wav"), x, ftm.constants["sr"])


# 4. Compute embeddings
models = ["LaionCLAP_audio", "LaionCLAP_music", "MSCLAP", "MERT_v1-95M", "MERT_v1-330M", "MERT_v0-public", "VGGish"]

for model_name in models:
    model = _load_model(model_name)
    embeddings_dir = f"exp_embeddings_linearity/generated/embeddings/{model_name}"
    os.makedirs(embeddings_dir, exist_ok=True)
    for i, couple in tqdm(enumerate(thetas_couples), desc=f"Processing {model_name}", total=len(thetas_couples)):
        audios = []
        audio = _load_audio(model, os.path.join(audio_dir, f"audio_row_{i}_A.wav"))
        embedding = model._get_embedding(audio)
        audio_embedding = torch.mean(embedding, dim=0).cpu().detach().numpy()
        np.save(os.path.join(embeddings_dir, f"embedding_{model_name}_row_{i}_A.npy"), audio_embedding)

        audio = _load_audio(model, os.path.join(audio_dir, f"audio_row_{i}_B.wav"))
        embedding = model._get_embedding(audio)
        audio_embedding = torch.mean(embedding, dim=0).cpu().detach().numpy()
        np.save(os.path.join(embeddings_dir, f"embedding_{model_name}_row_{i}_B.npy"), audio_embedding)

        range_AB = np.linspace(Theta_A, Theta_B, num=num_intermediate_samples, endpoint=False)[1:] # Do not resynthesize A
        for alpha, theta in enumerate(range_AB):
            audio = _load_audio(model, os.path.join(audio_dir, f"audio_row_{i}_AB_I{alpha}.wav"))
            audios.append(audio)
        
        for audio in audios:
            embedding = model._get_embedding(audio)
            audio_embedding = torch.mean(embedding, dim=0).cpu().detach().numpy()
            np.save(os.path.join(embeddings_dir, f"embedding_{model_name}_row_{i}_AB_I{alpha}.npy"), audio_embedding)

# 5. Compute metrics