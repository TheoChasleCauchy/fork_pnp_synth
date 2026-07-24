import torch
import numpy as np
import csv
from tqdm import tqdm

from exp_functions import embeddings

def process_csv(input_path: str):
    # Load trajectories tensor
    trajectories_tensor = torch.load(input_path)
    assert trajectories_tensor[0][0].shape == torch.Size(embeddings["cdpam"]), f"Expected shape {embeddings['cdpam']}, got {trajectories_tensor[0][0].shape}"
    
    # Compute smoothness value 
    smoothness_values = []
    for trajectory in tqdm(trajectories_tensor, desc="Computing Smoothness", total=len(trajectories_tensor)):
        dists = [torch.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]
        smoothness_mean, smoothness_std = torch.mean(torch.stack(dists)), torch.std(torch.stack(dists))
        smoothness_values.append((smoothness_mean.item(), smoothness_std.item()))

    return smoothness_values

def compute_smoothness_mean_cdpam(dirname: str, metric_csv: str, morph_type: str):
    # Load trajectories tensor
    trajectories_tensor_path = f"{dirname}/cdpam_{morph_type}_trajectories.pt"
    smoothness_values = process_csv(trajectories_tensor_path)

    # Write smoothness values in a csv file
    with open(metric_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Smoothness Mean CDPAM", "Smoothness Std CDPAM"])
        for i, (smoothness_mean, smoothness_std) in enumerate(smoothness_values):
            writer.writerow([i, smoothness_mean, smoothness_std])
        writer.writerow(["Mean Smoothness", f"{np.mean([mean for mean, _ in smoothness_values])} +- {np.std([mean for mean, _ in smoothness_values])}"])