import os
from tqdm import tqdm
import torch
import numpy as np

from load_models_and_audios import _load_audio, _load_model

def compute_trajectories_embeddings(models, trajectories, audio_dir, embeddings_dir):
    for model_name in models:
        model = _load_model(model_name)
        embeddings_dir = f"{embeddings_dir}/{model_name}"
        os.makedirs(embeddings_dir, exist_ok=True)
        for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Processing couples", total=len(trajectories))):
            for i_theta in range(len(trajectory)):
                audio = _load_audio(model, os.path.join(audio_dir, f"audio_row_{i_traj}_AB_I{i_theta}.wav"))
                embedding = model._get_embedding(audio)
                audio_embedding = torch.mean(embedding, dim=0).cpu().detach().numpy()
                np.save(os.path.join(embeddings_dir, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy"), audio_embedding)