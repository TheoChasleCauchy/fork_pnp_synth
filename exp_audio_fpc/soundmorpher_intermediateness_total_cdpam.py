import torch
import numpy as np
import csv, os
from tqdm import tqdm

from cdpam import lossnet_dfl
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def process_csv(trajectories: list[np.ndarray], embeddings_folder: str, model_name: str):
    
    cdpam_loss = lossnet_dfl(1024).to(device)

    # Compute intermediateness value 
    intermediateness_values = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing Intermediateness Total CDPAM", total=len(trajectories))):
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

        # intermediateness_value = torch.sum(torch.stack([torch.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]))
        intermediateness_value = torch.stack(CDPAM_values).sum()
        intermediateness_values.append(intermediateness_value.item())

    return intermediateness_values

def compute_intermediateness_total_cdpam(results_dir: str, model_name: str, trajectories: list[np.ndarray], embeddings_folder: str):

    intermediateness_values = process_csv(trajectories, embeddings_folder, model_name)

    # Write intermediateness values in a csv file
    with open(f"{results_dir}/{model_name}/{model_name}_intermediateness_total_cdpam_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Intermediateness Total CDPAM"])
        for i, intermediateness_value in enumerate(intermediateness_values):
            writer.writerow([i, intermediateness_value])
        writer.writerow(["Mean Intermediateness", f"{np.mean(intermediateness_values)} +- {np.std(intermediateness_values)}"])