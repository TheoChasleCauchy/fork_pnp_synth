import os, csv
import random
from generate_thetas import load_and_extract_couples
import numpy as np

# Hypercube: [[1500, 8000], [0.015, 1.0], [0.15, 2], [10**-5, 0.3], [0.25, 1.0]]
min_omega, max_omega = 1500, 8000
min_log_omega, max_log_omega = np.log10(min_omega), np.log10(max_omega)
min_tau, max_tau = 0.015, 1.0
min_logp, max_logp = np.log10(0.15), np.log10(2)
min_log_D, max_log_D = np.log10(10**-5), np.log10(0.3)
min_alpha, max_alpha = 0.25, 1.0


def generate_and_save_random_trajectories(seed, points_couples_filename, num_intermediate_samples, filename="exp_embeddings_linearity/generated/random_thetas_trajectories.csv"):

    random.seed(seed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    trajectories = []
    thetas_couples = load_and_extract_couples(points_couples_filename)
    for thetas_couple in thetas_couples:
        row = []
        row.extend(thetas_couple[0])

        for _ in range(num_intermediate_samples):
            log_omega = random.uniform(min_log_omega, max_log_omega)
            tau = random.uniform(min_tau, max_tau)
            log_p = random.uniform(min_logp, max_logp)
            log_D = random.uniform(min_log_D, max_log_D)
            alpha = random.uniform(min_alpha, max_alpha)
            row.extend([log_omega, tau, log_p, log_D, alpha])

        row.extend(thetas_couple[-1])

        trajectories.append(row)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filename
