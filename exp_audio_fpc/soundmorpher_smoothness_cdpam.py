import torch
import numpy as np
import csv, os
from tqdm import tqdm

from cdpam import lossnet_dfl
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def process_csv(trajectories: list[np.ndarray], embeddings_folder: str, model_name: str):

    cdpam_loss = lossnet_dfl(1024).to(device)

    # Compute smoothness value 
    smoothness_values = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing Smoothness CDPAM", total=len(trajectories))):
        morph_embeddings = []
        for i_theta in range(len(trajectory)):
            embedding = np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy"))
            embedding = torch.from_numpy(embedding).float()
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            morph_embeddings.append(embedding)
        
        CDPAM_values = []
        for i in range(len(morph_embeddings)-1):
            a1 = morph_embeddings[i].to(device)
            a1 = F.normalize(a1, dim=1)
            a2 = morph_embeddings[i+1].to(device)
            a2 = F.normalize(a2, dim=1)
            CDPAM_value = cdpam_loss.forward(a1,a2)
            CDPAM_values.append(CDPAM_value)

        smoothness_mean, smoothness_std = torch.mean(torch.stack(CDPAM_values)), torch.std(torch.stack(CDPAM_values))
        smoothness_values.append((smoothness_mean.item(), smoothness_std.item()))

    return smoothness_values

def compute_smoothness_mean_cdpam(results_dir: str, model_name: str, trajectories: list[np.ndarray], embeddings_folder: str):

    smoothness_values = process_csv(trajectories, embeddings_folder, model_name)

    # Write smoothness values in a csv file
    with open(f"{results_dir}/{model_name}/{model_name}_smoothness_mean_cdpam_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Smoothness Mean CDPAM", "Smoothness Std CDPAM"])
        for i, (smoothness_mean, smoothness_std) in enumerate(smoothness_values):
            writer.writerow([i, smoothness_mean, smoothness_std])
        writer.writerow(["Mean Smoothness", f"{np.mean([mean for mean, _ in smoothness_values])} +- {np.std([mean for mean, _ in smoothness_values])}"])