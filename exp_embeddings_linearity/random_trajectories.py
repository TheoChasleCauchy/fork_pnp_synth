import os, csv
import random

# Hypercube: [[1500, 8000], [0.015, 1.0], [0.15, 1000], [-5, -0.5], [0.01, 1.0]]
min_omega, max_omega = 1500, 8000
min_tau, max_tau = 0.015, 1.0
min_p, max_p = 0.15, 1000
min_D, max_D = -5, -0.5
min_alpha, max_alpha = 0.01, 1.0


def generate_and_save_random_trajectories(seed, number_of_couples, num_intermediate_samples, filename="exp_embeddings_linearity/generated/random_thetas_trajectories.csv"):

    random.seed(seed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    trajectories = []

    for _ in range(number_of_couples):

        thetas = []

        for _ in range(num_intermediate_samples+2):
            omega = random.uniform(min_omega, max_omega)
            tau = random.uniform(min_tau, max_tau)
            p = random.uniform(min_p, max_p)
            D = random.uniform(min_D, max_D)
            alpha = random.uniform(min_alpha, max_alpha)
            theta = (omega, tau, p, D, alpha)
            thetas.append(theta)

        # Flatten the list of thetas into a single row
        row = [val for theta in thetas for val in theta]
        trajectories.append(row)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{param}_{i}" for i in range(num_intermediate_samples+2) for param in ["omega", "tau", "p", "D", "alpha"]]
        writer.writerow(header)
        writer.writerows(trajectories)

    return filename
