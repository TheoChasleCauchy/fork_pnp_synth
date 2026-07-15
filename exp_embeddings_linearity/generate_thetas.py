import os
import random
import csv
import numpy as np

min_omega, max_omega = np.log10(40), np.log10(1000)
min_tau, max_tau = 0.4, 3
min_logp, max_logp = np.log10(10**-5), np.log10(0.2)
min_logD, max_logD = np.log10(10**-5), np.log10(0.3)
min_alpha, max_alpha = 10**-5, 1

def generate_and_save_couples(seed, number_of_couples, filename="exp_embeddings_linearity/generated/thetas_couples.csv"):

    random.seed(seed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    couples = []

    for _ in range(number_of_couples):
        # Sample A (logp = min_logp)
        omega_A = random.uniform(min_omega, max_omega)
        tau_A = random.uniform(min_tau, max_tau)
        logD_A = random.uniform(min_logD, max_logD)
        alpha_A = random.uniform(min_alpha, max_alpha)
        A = (omega_A, tau_A, min_logp, logD_A, alpha_A)

        # Sample B (logp = max_logp)
        omega_B = random.uniform(min_omega, max_omega)
        tau_B = random.uniform(min_tau, max_tau)
        logD_B = random.uniform(min_logD, max_logD)
        alpha_B = random.uniform(min_alpha, max_alpha)
        B = (omega_B, tau_B, max_logp, logD_B, alpha_B)

        # Flatten A and B into a single row
        row = (*A, *B)
        couples.append(row)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "omega_A", "tau_A", "logp_A", "logD_A", "alpha_A",
            "omega_B", "tau_B", "logp_B", "logD_B", "alpha_B"
        ]
        writer.writerow(header)
        writer.writerows(couples)

    return filename