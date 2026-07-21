import csv
import torch  # PyTorch for tensor operations
import numpy as np  # Library for numerical operations
import os
from tqdm import tqdm

def sobolev_distance(k: int, p: int, f, g, alpha_values):
    """
    Computes the Sobolev distance between two embeddings.

    Args:
        k (int): Order of the Sobolev space.
        p (int): Lp norm.
        f (list[torch.Tensor]): First function as list of embeddings.
        g (list[torch.Tensor]): Second function as list of embeddings.
        alpha_values (list[float]): List of alpha values for interpolation.

    Returns:
        float: The Sobolev distance (k=1, p=2) between the embeddings.
    """
    assert len(f) == len(g) == len(alpha_values), f"Length mismatch: {len(f)}, {len(g)}, {len(alpha_values)}"
    assert k in [0, 1], f"Unsupported Sobolev space order: {k}"
    
    terms_to_sum = []

    # First term
    k0 = torch.linalg.norm(torch.stack(f) - torch.stack(g), ord=p)
    terms_to_sum.append(k0)

    if k > 0:
        # Second term
        f_derivatives = []
        g_derivatives = []
        for i in range(len(f)):
            if i != len(f) - 1:
                f_prime = (f[i+1] - f[i]) / (alpha_values[i+1] - alpha_values[i])
                g_prime = (g[i+1] - g[i]) / (alpha_values[i+1] - alpha_values[i])
                
                f_derivatives.append(f_prime)
                g_derivatives.append(g_prime)
            else:
                # For the extremity (alpha = 1.0), copy the derivative of the previous point
                f_derivatives.append(f_prime)
                g_derivatives.append(g_prime)
        
        k1 = torch.linalg.norm(torch.stack(f_derivatives) - torch.stack(g_derivatives), ord=p)
        terms_to_sum.append(k1)
    
    dist = torch.sum(torch.stack(terms_to_sum))
    # dist = dist.pow(1/p) 

    return dist.item()

def compute_sobolev_distances(results_dir, model_name, trajectories):

    points_csv = f"exp_invalidate_metrics_example_1/generated/generated_intermediate_points.csv"
    
    trajectories = []
    with open(points_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            trajectories.append(row)
    
    for k, p in [(1, 2), (0, 2)]:
        sobolev_dists = []
        for trajectory in tqdm(trajectories, desc=f"Computing LCS on morphs", total=len(trajectories)):
            alpha_values = np.linspace(0, 1, len(trajectory))

            # Interpolation between vectors
            ideal_morphing = [torch.lerp(trajectory[0], trajectory[-1], alpha_value) for alpha_value in alpha_values]

            sobolev_value = sobolev_distance(k, p, trajectory, ideal_morphing, alpha_values = torch.tensor(alpha_values))
            sobolev_dists.append(sobolev_value)

        # Write sobolev values in a csv file
        with open(f"{results_dir}/{model_name}/{model_name}_sobolev_dists_{k}_{p}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Row", "Sobolev Distance"])
            for i, value in enumerate(sobolev_dists):
                writer.writerow([i, value])
            writer.writerow(["Mean Sobolev Distance", f"{np.mean(sobolev_dists)} +- {np.std(sobolev_dists)}"])