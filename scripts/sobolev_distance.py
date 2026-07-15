import argparse
import csv
import re
import torch  # PyTorch for tensor operations
import numpy as np  # Library for numerical operations
import os  # Library for interacting with the operating system

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
                # For the extremity (alpha = 0.5), copy the derivative of the previous point
                f_derivatives.append(f_prime)
                g_derivatives.append(g_prime)
        
        k1 = torch.linalg.norm(torch.stack(f_derivatives) - torch.stack(g_derivatives), ord=p)
        terms_to_sum.append(k1)
    
    dist = torch.sum(torch.stack(terms_to_sum))
    dist = dist.pow(1/p) 

    return dist.item()

def get_morphing_embeddings(embeddings_folder):

    npy_files = [f for f in os.listdir(embeddings_folder) if f.endswith(".npy")]

    parsed = []
    for f in npy_files:
        num_match = re.search(r'audio_(\d+)', f)
        val_match = re.search(r'value_(-?\d+\.?\d*)', f)
        if num_match and val_match:
            parsed.append((int(num_match.group(1)), float(val_match.group(1)), f))
    parsed.sort(key=lambda x: x[0])

    paths = [os.path.join(embeddings_folder, f) for _, _, f in parsed]
    values = [v for _, v, _ in parsed]

    # Load the embeddings from the numpy files
    embeddings = [torch.from_numpy(np.load(path)) for path in paths]

    return embeddings, values

def main(morphing_name):
    embeddings_folder = f"generations/{morphing_name}/embeddings"
    morphed_audios_embeddings, alpha_values = get_morphing_embeddings(embeddings_folder)

    # get alpha values to b between 0 and 1
    alpha_values = torch.tensor(alpha_values)
    alpha_values = (alpha_values - alpha_values.min()) / (alpha_values.max() - alpha_values.min())

    # Interpolation between vectors
    ideal_morphing = [torch.lerp(morphed_audios_embeddings[0], morphed_audios_embeddings[-1], alpha) for alpha in alpha_values]

    for k, p in [(1, 2), (0, 2)]:
        sobolev_dist = sobolev_distance(k, p, morphed_audios_embeddings, ideal_morphing, alpha_values)

        # Write the values in a csv file
        with open(f"generations/{morphing_name}/{morphing_name}_sobolev_distance_{k}_{p}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Sobolev Distance"])
            writer.writerow([sobolev_dist])

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load the morphing's audios.")
    parser.add_argument("morphing_name", type=str, help="Name of the morphing")

    # Parse arguments
    args = parser.parse_args()

    # Call main with the directory argument
    main(args.morphing_name)