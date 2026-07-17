import os
import random
import csv
import numpy as np
from pyparsing import alphas

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
        omega = random.uniform(min_omega, max_omega)
        tau = random.uniform(min_tau, max_tau)
        logD = random.uniform(min_logD, max_logD)
        alpha = random.uniform(min_alpha, max_alpha)
        A = (omega, tau, min_logp, logD, alpha)
        B = (omega, tau, max_logp, logD, alpha)

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

def save_logp_values(num_intermediate_samples, filename="exp_embeddings_linearity/generated/logp_values.csv"):
    logp_values = np.linspace(min_logp, max_logp, num=num_intermediate_samples, endpoint=False)
    logp_values = list(logp_values) + [max_logp]

    # Save csv file:
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(logp_values)

    return filename