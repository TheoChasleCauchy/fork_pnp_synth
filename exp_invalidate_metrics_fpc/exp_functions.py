import os

import numpy as np
import csv

def sample_couples_2D_space(n_couples, filename, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    couples = []
    for _ in range(n_couples):
        a = np.random.rand(2) * 10  # Random point in 2D space
        b = np.random.rand(2) * 10  # Random point in 2D space
        couples.append([a[0], a[1], b[0], b[1]])
    
    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "A_X",
            "A_Y",
            "B_X",
            "B_Y"
        ]
        writer.writerow(header)
        writer.writerows(couples)
    
    return couples

def create_intermediate_points(couples, num_intermediate_samples, filename):
    trajectories_points = []
    for a_x, a_y, b_x, b_y in couples:
        a = np.array([a_x, a_y])
        b = np.array([b_x, b_y])
        trajectory_points = []
        trajectory_points.extend(a)
        # Create intermediate points on the circle
        for i in range(1, num_intermediate_samples+1):
            random_angle = np.random.rand(1) * 2 * np.pi
            distance_from_a = np.linalg.norm(a - b) * i / (num_intermediate_samples+2)
            x = a_x + distance_from_a * np.cos(random_angle)
            y = a_y + distance_from_a * np.sin(random_angle)
            trajectory_points.extend([x[0], y[0]])
            
        trajectory_points.extend(b)
        
        trajectories_points.append(trajectory_points)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "A_X", "A_Y", *[f"P{i}_X" for i in range(num_intermediate_samples)], *[f"P{i}_Y" for i in range(num_intermediate_samples)], "B_X", "B_Y"
        ]
        writer.writerow(header)
        for trajectory in trajectories_points:
            writer.writerow(trajectory)
    
    return trajectories_points

def create_linear_intermediate_points(couples, num_intermediate_samples, filename):
    trajectories_points = []
    for a_x, a_y, b_x, b_y in couples:
        a = np.array([a_x, a_y])
        b = np.array([b_x, b_y])
        trajectory_points = []
        trajectory_points.extend(a)
        # Create intermediate points on the line segment
        for i in range(1, num_intermediate_samples+1):
            alpha = i / (num_intermediate_samples+1)
            x = a_x + alpha * (b_x - a_x)
            y = a_y + alpha * (b_y - a_y)
            trajectory_points.extend([x, y])
            
        trajectory_points.extend(b)
        
        trajectories_points.append(trajectory_points)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "A_X", "A_Y", *[f"P{i}_X" for i in range(num_intermediate_samples)], *[f"P{i}_Y" for i in range(num_intermediate_samples)], "B_X", "B_Y"
        ]
        writer.writerow(header)
        for trajectory in trajectories_points:
            writer.writerow(trajectory)
    
    return trajectories_points

def create_random_intermediate_points(couples, num_intermediate_samples, filename):
    trajectories_points = []
    for a_x, a_y, b_x, b_y in couples:
        a = np.array([a_x, a_y])
        b = np.array([b_x, b_y])
        trajectory_points = []
        trajectory_points.extend(a)
        # Create random intermediate points in the whole 2D space
        for i in range(1, num_intermediate_samples+1):
            x = np.random.rand(1) * 10  # Random point in 2D space
            y = np.random.rand(1) * 10  # Random point in 2D space
            trajectory_points.extend([x.item(), y.item()])
            
        trajectory_points.extend(b)
        
        trajectories_points.append(trajectory_points)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "A_X", "A_Y", *[f"P{i}_X" for i in range(num_intermediate_samples)], *[f"P{i}_Y" for i in range(num_intermediate_samples)], "B_X", "B_Y"
        ]
        writer.writerow(header)
        for trajectory in trajectories_points:
            writer.writerow(trajectory)
    
    return trajectories_points

# Make a table of the mean result of each metric
def make_table():
    import re
    
    def get_metrics_values(results_dir):
        metrics_values = {}

        # Get lcs metric value
        lcs_csv_path = os.path.join(results_dir, "mix2morph_lcs_values.csv")
        with open(lcs_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_lcs = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_lcs, std_lcs = map(float, mean_std_lcs)
            metrics_values["LCS"] = (mean_lcs, std_lcs)
        
        # Get Smoothness Clap metric value
        clap_csv_path = os.path.join(results_dir, "smoothness_clap_corr_values.csv")
        with open(clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["Smoothness CLAP"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)
        
        # Get Sobolev k=0, p=2 value
        sobolev_k0_p2_csv_path = os.path.join(results_dir, "sobolev_dists_0_2.csv")
        with open(sobolev_k0_p2_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_sobolev_k0_p2 = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_sobolev_k0_p2, std_sobolev_k0_p2 = map(float, mean_std_sobolev_k0_p2)
            metrics_values["Sobolev (0, 2)"] = (mean_sobolev_k0_p2, std_sobolev_k0_p2)
        
        # Get Sobolev k=1, p=2 value
        sobolev_k1_p2_csv_path = os.path.join(results_dir, "sobolev_dists_1_2.csv")
        with open(sobolev_k1_p2_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_sobolev_k1_p2 = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_sobolev_k1_p2, std_sobolev_k1_p2 = map(float, mean_std_sobolev_k1_p2)
            metrics_values["Sobolev (1, 2)"] = (mean_sobolev_k1_p2, std_sobolev_k1_p2)
        
        # Get Correspondence value
        correspondence_csv_path = os.path.join(results_dir, "soundmorpher_correspondence_mfccs_values.csv")
        with open(correspondence_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
            metrics_values["Correspondence"] = (mean_correspondence, std_correspondence)
        
        # Get Intermediateness value
        intermediateness_csv_path = os.path.join(results_dir, "intermediateness_total_cdpam_values.csv")
        with open(intermediateness_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
            metrics_values["Intermediateness"] = (mean_intermediateness, std_intermediateness)
        
        # Get Smoothness CDPAM value
        smoothness_cdpam_csv_path = os.path.join(results_dir, "smoothness_mean_cdpam_values.csv")
        with open(smoothness_cdpam_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
            metrics_values["Smoothness CDPAM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)
        
        return metrics_values
        
    results_dir_fpc_1 = "exp_invalidate_metrics_fpc/generated/metrics/fpc_1"
    fpc_1_metrics_values = get_metrics_values(results_dir_fpc_1)
    results_dir_linear = "exp_invalidate_metrics_fpc/generated/metrics/linear"
    linear_metrics_values = get_metrics_values(results_dir_linear)
    results_dir_random = "exp_invalidate_metrics_fpc/generated/metrics/random"
    random_metrics_values = get_metrics_values(results_dir_random)

    # Write the table to a CSV file
    results_dir = "exp_invalidate_metrics_fpc/generated/metrics"
    output_csv_path = os.path.join(results_dir, "metrics_table.csv")
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header: metrics as rows
        # header = ["Sobolev Distance (0, 2) $\downarrow$", "Sobolev Distance (1, 2) $\downarrow$"]
        header = ["Metric", "Ideal Value", "FPC 1 Value", "Random Value"]
        writer.writerow(header)

        # Write rows: models as rows, (k, p) as columns, mean+-std as values
        row = [
            "LCS",
            f"{linear_metrics_values['LCS'][0]:.2f} +- {linear_metrics_values['LCS'][1]:.2f}",
            f"{fpc_1_metrics_values['LCS'][0]:.2f} +- {fpc_1_metrics_values['LCS'][1]:.2f}",
            f"{random_metrics_values['LCS'][0]:.2f} +- {random_metrics_values['LCS'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            f"{linear_metrics_values['Smoothness CLAP'][0]:.2f} +- {linear_metrics_values['Smoothness CLAP'][1]:.2f}",
            f"{fpc_1_metrics_values['Smoothness CLAP'][0]:.2f} +- {fpc_1_metrics_values['Smoothness CLAP'][1]:.2f}",
            f"{random_metrics_values['Smoothness CLAP'][0]:.2f} +- {random_metrics_values['Smoothness CLAP'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Correspondence",
            f"{linear_metrics_values['Correspondence'][0]:.2f} +- {linear_metrics_values['Correspondence'][1]:.2f}",
            f"{fpc_1_metrics_values['Correspondence'][0]:.2f} +- {fpc_1_metrics_values['Correspondence'][1]:.2f}",
            f"{random_metrics_values['Correspondence'][0]:.2f} +- {random_metrics_values['Correspondence'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            f"{linear_metrics_values['Intermediateness'][0]:.2f} +- {linear_metrics_values['Intermediateness'][1]:.2f}",
            f"{fpc_1_metrics_values['Intermediateness'][0]:.2f} +- {fpc_1_metrics_values['Intermediateness'][1]:.2f}",
            f"{random_metrics_values['Intermediateness'][0]:.2f} +- {random_metrics_values['Intermediateness'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CDPAM",
            f"{linear_metrics_values['Smoothness CDPAM'][0]:.2f} +- {linear_metrics_values['Smoothness CDPAM'][1]:.2f}",
            f"{fpc_1_metrics_values['Smoothness CDPAM'][0]:.2f} +- {fpc_1_metrics_values['Smoothness CDPAM'][1]:.2f}",
            f"{random_metrics_values['Smoothness CDPAM'][0]:.2f} +- {random_metrics_values['Smoothness CDPAM'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (0, 2)",
            f"{linear_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{fpc_1_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {fpc_1_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (0, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (1, 2)",
            f"{linear_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{fpc_1_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {fpc_1_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (1, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
    