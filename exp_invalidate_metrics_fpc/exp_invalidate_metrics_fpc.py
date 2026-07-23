import os
import numpy as np
import matplotlib.pyplot as plt

from sobolev_distance import compute_sobolev_distances
from exp_functions import sample_couples_specified_dimensions_space, create_fpc1_intermediate_points, create_fpc2_intermediate_points, create_fpc3_intermediate_points, create_linear_intermediate_points, create_random_intermediate_points, make_table
from soundmorpher_correspondence_mfccs import compute_soundmorpher_correspondence_mfccs
from mix2morph_lcs import compute_mix2morph_lcs
from soundmorpher_smoothness_cdpam import compute_smoothness_mean_cdpam 
from soundmorpher_intermediateness_total_cdpam import compute_intermediateness_total_cdpam
from morphfader_smoothness_clap_corr import compute_smoothness_clap_corr

seed = 42
np.random.seed(seed)

def plot_trajectory_points(trajectory, output_path="exp_invalidate_metrics_fpc/generated/trajectories.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure()

    x_coords = [trajectory[i] for i in range(0, len(trajectory), 2)]
    y_coords = [trajectory[i+1] for i in range(0, len(trajectory), 2)]
    tab10_colors = plt.cm.tab10.colors
    plt.scatter(x_coords[0], y_coords[0], color=tab10_colors[0], s=20)  # s=10 sets the point size
    plt.scatter(x_coords[1:-1], y_coords[1:-1], color=tab10_colors[1], s=10)  # s=10 sets the point size
    plt.scatter(x_coords[-1], y_coords[-1], color=tab10_colors[2], s=20)  # s=10 sets the point size

    # Annotate each point with its index
    plt.annotate("A", (x_coords[0], y_coords[0]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    for j, (x, y) in enumerate(zip(x_coords[1:-1], y_coords[1:-1])):
        plt.annotate(str(j), (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)
    plt.annotate("B", (x_coords[-1], y_coords[-1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    plt.grid(True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

number_of_couples = 1000
num_intermediate_samples = 11 # Need to be odd for Correspondence and LCS as their use only the middle morph sample
dimensions = 512

# 1. Prendre 1000 paires (A,B) de points aléatoires dans un plan 2D.
os.makedirs("exp_invalidate_metrics_fpc/generated/", exist_ok=True)
couples_filename = "exp_invalidate_metrics_fpc/generated/generated_couples.csv"
couples = sample_couples_specified_dimensions_space(number_of_couples, couples_filename, dimensions=dimensions)

# FPC 1
print("Creating FPC 1 intermediate points...")
# 2. Placer les points intermédiaires de chaque trajectoire AB sur un cercle de rayon alpha * AB avec un angle aléatoire.
intermediate_points_filename = "exp_invalidate_metrics_fpc/generated/generated_fpc1_intermediate_points.csv"
trajectories = create_fpc1_intermediate_points(couples, num_intermediate_samples, intermediate_points_filename, dimensions=dimensions)

# Optional: visualize the points
visualize = False
if visualize:
    plot_trajectory_points(trajectories[0], output_path="exp_invalidate_metrics_fpc/generated/visualizations/fpc1.png")

# # 3. Calculer métriques sur ces trajectoires.
metrics_folder = "exp_invalidate_metrics_fpc/generated/metrics/fpc1"
os.makedirs(metrics_folder, exist_ok=True)
points_csv = f"exp_invalidate_metrics_fpc/generated/generated_fpc1_intermediate_points.csv"
compute_mix2morph_lcs(points_csv, f"{metrics_folder}/mix2morph_lcs_values.csv", dimensions=dimensions)
compute_intermediateness_total_cdpam(points_csv, f"{metrics_folder}/intermediateness_total_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_mean_cdpam(points_csv, f"{metrics_folder}/smoothness_mean_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_clap_corr(points_csv, f"{metrics_folder}/smoothness_clap_corr_values.csv", dimensions=dimensions)
compute_soundmorpher_correspondence_mfccs(points_csv, f"{metrics_folder}/soundmorpher_correspondence_mfccs_values.csv", dimensions=dimensions)
compute_sobolev_distances(points_csv, f"{metrics_folder}", dimensions=dimensions)

# Baseline linéaire
print("Creating linear intermediate points...")
# 4. Placer les points intermédiaires uniformément sur le segment [AB].
intermediate_points_filename = "exp_invalidate_metrics_fpc/generated/generated_linear_intermediate_points.csv"
trajectories = create_linear_intermediate_points(couples, num_intermediate_samples, intermediate_points_filename, dimensions=dimensions)

visualize = False
if visualize:
    plot_trajectory_points(trajectories[0], output_path="exp_invalidate_metrics_fpc/generated/visualizations/linear.png")

# 5. Calculer métriques sur ces trajectoires.
metrics_folder = "exp_invalidate_metrics_fpc/generated/metrics/linear"
os.makedirs(metrics_folder, exist_ok=True)
points_csv = f"exp_invalidate_metrics_fpc/generated/generated_linear_intermediate_points.csv"
compute_mix2morph_lcs(points_csv, f"{metrics_folder}/mix2morph_lcs_values.csv", dimensions=dimensions)
compute_intermediateness_total_cdpam(points_csv, f"{metrics_folder}/intermediateness_total_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_mean_cdpam(points_csv, f"{metrics_folder}/smoothness_mean_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_clap_corr(points_csv, f"{metrics_folder}/smoothness_clap_corr_values.csv", dimensions=dimensions)
compute_soundmorpher_correspondence_mfccs(points_csv, f"{metrics_folder}/soundmorpher_correspondence_mfccs_values.csv", dimensions=dimensions)
compute_sobolev_distances(points_csv, f"{metrics_folder}", dimensions=dimensions)

# Baseline aléatoire
print("Creating random intermediate points...")
# 4. Placer les points intermédiaires aléatoirement dans le plan.
intermediate_points_filename = "exp_invalidate_metrics_fpc/generated/generated_random_intermediate_points.csv"
trajectories = create_random_intermediate_points(couples, num_intermediate_samples, intermediate_points_filename, dimensions=dimensions)

visualize = False
if visualize:
    plot_trajectory_points(trajectories[0], output_path="exp_invalidate_metrics_fpc/generated/visualizations/random.png")

# 5. Calculer métriques sur ces trajectoires.
metrics_folder = "exp_invalidate_metrics_fpc/generated/metrics/random"
os.makedirs(metrics_folder, exist_ok=True)
points_csv = f"exp_invalidate_metrics_fpc/generated/generated_random_intermediate_points.csv"
compute_mix2morph_lcs(points_csv, f"{metrics_folder}/mix2morph_lcs_values.csv", dimensions=dimensions)
compute_intermediateness_total_cdpam(points_csv, f"{metrics_folder}/intermediateness_total_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_mean_cdpam(points_csv, f"{metrics_folder}/smoothness_mean_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_clap_corr(points_csv, f"{metrics_folder}/smoothness_clap_corr_values.csv", dimensions=dimensions)
compute_soundmorpher_correspondence_mfccs(points_csv, f"{metrics_folder}/soundmorpher_correspondence_mfccs_values.csv", dimensions=dimensions)
compute_sobolev_distances(points_csv, f"{metrics_folder}", dimensions=dimensions)

print("Creating FPC 2 intermediate points...")
# FPC 2
# Placer les num_intermediate_samples-1 premiers points proche de 1 et le dernier proche de B.
intermediate_points_filename = "exp_invalidate_metrics_fpc/generated/generated_fpc2_intermediate_points.csv"
trajectories = create_fpc2_intermediate_points(couples, num_intermediate_samples, intermediate_points_filename, dimensions=dimensions)

visualize = False
if visualize:
    plot_trajectory_points(trajectories[0], output_path="exp_invalidate_metrics_fpc/generated/visualizations/fpc2.png")

# Calculer métriques sur ces trajectoires.
metrics_folder = "exp_invalidate_metrics_fpc/generated/metrics/fpc2"
os.makedirs(metrics_folder, exist_ok=True)
points_csv = f"exp_invalidate_metrics_fpc/generated/generated_fpc2_intermediate_points.csv"
compute_mix2morph_lcs(points_csv, f"{metrics_folder}/mix2morph_lcs_values.csv", dimensions=dimensions)
compute_intermediateness_total_cdpam(points_csv, f"{metrics_folder}/intermediateness_total_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_mean_cdpam(points_csv, f"{metrics_folder}/smoothness_mean_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_clap_corr(points_csv, f"{metrics_folder}/smoothness_clap_corr_values.csv", dimensions=dimensions)
compute_soundmorpher_correspondence_mfccs(points_csv, f"{metrics_folder}/soundmorpher_correspondence_mfccs_values.csv", dimensions=dimensions)
compute_sobolev_distances(points_csv, f"{metrics_folder}", dimensions=dimensions)

print("Creating FPC 3 intermediate points...")
# FPC 3
# Placer les num_intermediate_samples-1 premiers points proche de 1 et le dernier proche de B.
intermediate_points_filename = "exp_invalidate_metrics_fpc/generated/generated_fpc3_intermediate_points.csv"
trajectories = create_fpc3_intermediate_points(couples, num_intermediate_samples, intermediate_points_filename, dimensions=dimensions)

visualize = False
if visualize:
    plot_trajectory_points(trajectories[0], output_path="exp_invalidate_metrics_fpc/generated/visualizations/fpc3.png")

# 5. Calculer métriques sur ces trajectoires.
metrics_folder = "exp_invalidate_metrics_fpc/generated/metrics/fpc3"
os.makedirs(metrics_folder, exist_ok=True)
points_csv = f"exp_invalidate_metrics_fpc/generated/generated_fpc3_intermediate_points.csv"
compute_mix2morph_lcs(points_csv, f"{metrics_folder}/mix2morph_lcs_values.csv", dimensions=dimensions)
compute_intermediateness_total_cdpam(points_csv, f"{metrics_folder}/intermediateness_total_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_mean_cdpam(points_csv, f"{metrics_folder}/smoothness_mean_cdpam_values.csv", dimensions=dimensions)
compute_smoothness_clap_corr(points_csv, f"{metrics_folder}/smoothness_clap_corr_values.csv", dimensions=dimensions)
compute_soundmorpher_correspondence_mfccs(points_csv, f"{metrics_folder}/soundmorpher_correspondence_mfccs_values.csv", dimensions=dimensions)
compute_sobolev_distances(points_csv, f"{metrics_folder}", dimensions=dimensions)

# 6. Make a table
make_table()