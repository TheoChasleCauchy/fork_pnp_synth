import csv
import torch  # PyTorch for tensor operations
import numpy as np  # Library for numerical operations
import os
from tqdm import tqdm

embeddings_sizes = {
    "LaionCLAP_audio": 512,
    "LaionCLAP_music": 512,
    "MSCLAP": 1024,
    "MERT_v1-95M": 768,
    "MERT_v1-330M": 768,
    "MERT_v0-public": 768,
    "VGGish": 128
}

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

def compute_sobolev_distances(embeddings_folder, results_dir, model_name, trajectories, num_intermediate_samples):

    # get alpha values to b between 0 and 1
    p_values = torch.tensor(np.linspace(0, 1, num_intermediate_samples+2))
    
    for k, p in [(1, 2), (0, 2)]:
        sobolev_dists = []
        for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Processing couples", total=len(trajectories))):
            for i_theta in range(len(trajectory)):
                morph_embeddings = []
                embedding = torch.tensor(np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy")))
                
                # Normalize to 1.0
                embedding = embedding / torch.norm(embedding)

                # Normalize by embedding size
                embedding = embedding / embeddings_sizes[model_name]

                morph_embeddings.append(embedding)

            # Interpolation between vectors
            ideal_morphing = [torch.lerp(morph_embeddings[0], morph_embeddings[-1], p_value) for p_value in p_values]

            sobolev_value = sobolev_distance(k, p, morph_embeddings, ideal_morphing, alpha_values = p_values)
            sobolev_dists.append(sobolev_value)

        # Write sobolev values in a csv file
        with open(f"{results_dir}/{model_name}/{model_name}_sobolev_dists_{k}_{p}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Row", "Sobolev Distance"])
            for i, value in enumerate(sobolev_dists):
                writer.writerow([i, value])
            writer.writerow(["Mean Sobolev Distance", f"{np.mean(sobolev_dists)} +- {np.std(sobolev_dists)}"])
            
def make_table(results_dir, models):
    # Initialize a dictionary to store the mean and std for each (k, p) and model
    table_data = {}

    for k, p in [(1, 2), (0, 2)]:
        row_key = f"Sobolev ({k}, {p}) $\downarrow$"
        table_data[row_key] = {}

        for model_name in models:
            csv_path = os.path.join(results_dir, model_name, f"{model_name}_sobolev_dists_{k}_{p}.csv")
            with open(csv_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                # Extract the last row, which contains the mean and std
                mean_std_str = rows[-1][1]
                table_data[row_key][model_name] = mean_std_str

    # Write the table to a CSV file
    output_csv_path = os.path.join(results_dir, "sobolev_distances_table.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header: models as rows
        header = ["Model", "Sobolev Distance (0, 2) $\downarrow$", "Sobolev Distance (1, 2) $\downarrow$", "Size"]
        writer.writerow(header)

        # Write rows: models as rows, (k, p) as columns, mean+-std as values
        for model_name in models:
            row = [model_name]
            for k, p in [(0, 2), (1, 2)]:
                row.append(table_data[f"Sobolev ({k}, {p}) $\downarrow$"][model_name])
            row.append(embeddings_sizes[model_name])
            writer.writerow(row)