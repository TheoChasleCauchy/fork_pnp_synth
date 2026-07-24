import torch
import numpy as np
import csv
from tqdm import tqdm

from exp_functions import embeddings

def process_csv(input_path: str):
    # Load trajectories tensor
    trajectories_tensor = torch.load(input_path)
    assert trajectories_tensor[0][0].shape == torch.Size(embeddings["cdpam"]), f"Expected shape {embeddings['cdpam']}, got {trajectories_tensor[0][0].shape}"

    # Compute intermediateness value 
    intermediateness_values = []
    for trajectory in tqdm(trajectories_tensor, desc="Computing Intermediateness", total=len(trajectories_tensor)):
        intermediateness_value = torch.sum(torch.stack([torch.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]))
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