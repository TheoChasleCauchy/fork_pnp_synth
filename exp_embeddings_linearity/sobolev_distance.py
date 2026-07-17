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

def compute_sobolev_distances(embeddings_folder, results_dir, model_name, thetas_couples, logp_values, num_intermediate_samples):

    # get alpha values to b between 0 and 1
    logp_values = torch.tensor(logp_values)
    logp_values = (logp_values - logp_values.min()) / (logp_values.max() - logp_values.min())
    
    for k, p in [(1, 2), (0, 2)]:
        sobolev_dists = []
        for i in tqdm(range(len(thetas_couples)), desc=f"Applying Sobolev ({k},{p}) to {model_name}", total=len(thetas_couples)):
            
            morph_embeddings = []
            
            embedding_A = torch.tensor(np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i}_A.npy")))
            morph_embeddings.append(embedding_A)
            embedding_B = torch.tensor(np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i}_B.npy")))
            morph_embeddings.append(embedding_B)
    
            for alpha in range(num_intermediate_samples-1):
                embedding = torch.tensor(np.load(os.path.join(embeddings_folder, f"embedding_{model_name}_row_{i}_AB_I{alpha}.npy")))
                morph_embeddings.append(embedding)

            # Interpolation between vectors
            ideal_morphing = [torch.lerp(embedding_A, embedding_B, logp_value) for logp_value in logp_values]

            sobolev_value = sobolev_distance(k, p, morph_embeddings, ideal_morphing, alpha_values = logp_values)
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

        # Write header: models as columns
        header = ["Sobolev Distance (k, p)"] + models
        writer.writerow(header)

        # Write rows: (k, p) as rows, mean+-std as values
        for row_key, model_data in table_data.items():
            row = [row_key] + [model_data[model] for model in models]
            writer.writerow(row)