import os

from sobolev_distance import compute_sobolev_distances
from exp_functions import sample_couples_2D_space, create_intermediate_points, create_linear_intermediate_points, create_random_intermediate_points, make_table
from soundmorpher_correspondence_mfccs import compute_soundmorpher_correspondence_mfccs
from mix2morph_lcs import compute_mix2morph_lcs
from soundmorpher_smoothness_cdpam import compute_smoothness_mean_cdpam 
from soundmorpher_intermediateness_total_cdpam import compute_intermediateness_total_cdpam
from morphfader_smoothness_clap_corr import compute_smoothness_clap_corr

seed = 42

number_of_couples = 1000
num_intermediate_samples = 10
dimensions = 512

# 1. Prendre 1000 paires (A,B) de points aléatoires dans un plan 2D.
os.makedirs("exp_invalidate_metrics_fpc/generated/", exist_ok=True)
couples_filename = "exp_invalidate_metrics_fpc/generated/generated_couples.csv"
couples = sample_couples_2D_space(number_of_couples, couples_filename, seed=seed)

# 2. Placer les points intermédiaires de chaque trajectoire AB sur un cercle de rayon alpha * AB avec un angle aléatoire.
intermediate_points_filename = "exp_invalidate_metrics_fpc/generated/generated_intermediate_points.csv"
trajectories = create_intermediate_points(couples, num_intermediate_samples, intermediate_points_filename)

# Optional: visualize the points
visualize = False
if visualize:
    import matplotlib.pyplot as plt
    def plot_trajectory_points(trajectory, output_path="exp_invalidate_metrics_fpc/generated/trajectories.png"):
        plt.figure()

        x_coords = [point[0] for point in trajectory]
        y_coords = [point[1] for point in trajectory]
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

    plot_trajectory_points(trajectories[0])

# 3. Calculer métriques sur ces trajectoires.
metrics_folder = "exp_invalidate_metrics_fpc/generated/metrics/fpc_1"
os.makedirs(metrics_folder, exist_ok=True)
points_csv = f"exp_invalidate_metrics_fpc/generated/generated_intermediate_points.csv"
compute_mix2morph_lcs(points_csv, f"{metrics_folder}/mix2morph_lcs_values.csv")
compute_intermediateness_total_cdpam(points_csv, f"{metrics_folder}/intermediateness_total_cdpam_values.csv")
compute_smoothness_mean_cdpam(points_csv, f"{metrics_folder}/smoothness_mean_cdpam_values.csv")
compute_smoothness_clap_corr(points_csv, f"{metrics_folder}/smoothness_clap_corr_values.csv")
compute_soundmorpher_correspondence_mfccs(points_csv, f"{metrics_folder}/soundmorpher_correspondence_mfccs_values.csv")
compute_sobolev_distances(points_csv, f"{metrics_folder}")

# Baseline linéaire
# 4. Placer les points intermédiaires uniformément sur le segment [AB].
intermediate_points_filename = "exp_invalidate_metrics_fpc/generated/generated_linear_intermediate_points.csv"
trajectories = create_linear_intermediate_points(couples, num_intermediate_samples, intermediate_points_filename)

# 5. Calculer métriques sur ces trajectoires.
metrics_folder = "exp_invalidate_metrics_fpc/generated/metrics/linear"
os.makedirs(metrics_folder, exist_ok=True)
points_csv = f"exp_invalidate_metrics_fpc/generated/generated_linear_intermediate_points.csv"
compute_mix2morph_lcs(points_csv, f"{metrics_folder}/mix2morph_lcs_values.csv")
compute_intermediateness_total_cdpam(points_csv, f"{metrics_folder}/intermediateness_total_cdpam_values.csv")
compute_smoothness_mean_cdpam(points_csv, f"{metrics_folder}/smoothness_mean_cdpam_values.csv")
compute_smoothness_clap_corr(points_csv, f"{metrics_folder}/smoothness_clap_corr_values.csv")
compute_soundmorpher_correspondence_mfccs(points_csv, f"{metrics_folder}/soundmorpher_correspondence_mfccs_values.csv")
compute_sobolev_distances(points_csv, f"{metrics_folder}")

# Baseline aléatoire
# 4. Placer les points intermédiaires aléatoirement dans le plan.
intermediate_points_filename = "exp_invalidate_metrics_fpc/generated/generated_random_intermediate_points.csv"
trajectories = create_random_intermediate_points(couples, num_intermediate_samples, intermediate_points_filename)

# 5. Calculer métriques sur ces trajectoires.
metrics_folder = "exp_invalidate_metrics_fpc/generated/metrics/random"
os.makedirs(metrics_folder, exist_ok=True)
points_csv = f"exp_invalidate_metrics_fpc/generated/generated_random_intermediate_points.csv"
compute_mix2morph_lcs(points_csv, f"{metrics_folder}/mix2morph_lcs_values.csv")
compute_intermediateness_total_cdpam(points_csv, f"{metrics_folder}/intermediateness_total_cdpam_values.csv")
compute_smoothness_mean_cdpam(points_csv, f"{metrics_folder}/smoothness_mean_cdpam_values.csv")
compute_smoothness_clap_corr(points_csv, f"{metrics_folder}/smoothness_clap_corr_values.csv")
compute_soundmorpher_correspondence_mfccs(points_csv, f"{metrics_folder}/soundmorpher_correspondence_mfccs_values.csv")
compute_sobolev_distances(points_csv, f"{metrics_folder}")

# 6. Make a table
make_table()