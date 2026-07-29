import torch
import numpy as np
import csv
from tqdm import tqdm

from exp_functions import embeddings
from cdpam import CDPAM
import torch.nn.functional as F

model = CDPAM(dev='cuda:0' if torch.cuda.is_available() else 'cpu')

def process_csv(input_path: str):
    # Load trajectories tensor
    trajectories_tensor = torch.load(input_path)
    assert trajectories_tensor[0][0].shape == torch.Size(embeddings["cdpam"]), f"Expected shape {embeddings['cdpam']}, got {trajectories_tensor[0][0].shape}"

    # Compute intermediateness value 
    intermediateness_values = []
    for trajectory in tqdm(trajectories_tensor, desc="Computing Intermediateness", total=len(trajectories_tensor)):
        CDPAM_values = []
        for i in range(len(trajectory)-1):
            a1 = trajectory[i].to(model.device)
            a1 = F.normalize(a1, dim=1)
            a2 = trajectory[i+1].to(model.device)
            a2 = F.normalize(a2, dim=1)
            CDPAM_value = model.model.model_dist.forward(a1,a2)
            CDPAM_values.append(CDPAM_value)

        # intermediateness_value = torch.sum(torch.stack([torch.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]))
        intermediateness_value = torch.stack(CDPAM_values).sum()
        intermediateness_values.append(intermediateness_value.item())

    return intermediateness_values

def compute_intermediateness_total_cdpam(dirname: str, metric_csv: str, morph_type: str):
    # Load trajectories tensor
    trajectories_tensor_path = f"{dirname}/cdpam_{morph_type}_trajectories.pt"
    intermediateness_values = process_csv(trajectories_tensor_path)

    # Write intermediateness values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Intermediateness Total CDPAM"])
        for i, intermediateness_value in enumerate(intermediateness_values):
            writer.writerow([i, intermediateness_value])
        writer.writerow(["Mean Intermediateness", f"{np.mean(intermediateness_values)} +- {np.std(intermediateness_values)}"])