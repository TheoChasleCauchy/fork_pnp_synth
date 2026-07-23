import os

import numpy as np
import csv

def sample_couples_specified_dimensions_space(n_couples, filename, dimensions=2):    
    couples = []
    for _ in range(n_couples):
        a = np.random.rand(dimensions) * 10  # Random point in specified dimensions space
        b = np.random.rand(dimensions) * 10  # Random point in specified dimensions space
        couples.append(list(a) + list(b))
    
    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(dimensions):
            header.append(f"A_{i}")
        for i in range(dimensions):
            header.append(f"B_{i}")
        
        writer.writerow(header)
        writer.writerows(couples)
    
    return couples

def create_fpc1_intermediate_points(couples, num_intermediate_samples, filename, dimensions=2):
    trajectories_points = []
    for couple in couples:
        a = np.array(couple[:dimensions])
        b = np.array(couple[dimensions:2*dimensions])

        trajectory_points = a.tolist()

        for i in range(1, num_intermediate_samples + 1):
            random_direction = np.random.randn(dimensions)
            random_direction /= np.linalg.norm(random_direction)  # Normalize to unit vector

            distance_from_a = np.linalg.norm(a - b) * i / (num_intermediate_samples + 2)
            intermediate_point = a + distance_from_a * random_direction
            trajectory_points.extend(intermediate_point.tolist())

        trajectory_points.extend(b.tolist())
        trajectories_points.append(trajectory_points)

    # Generate header dynamically
    header = []
    header.extend([f"A_{i}" for i in range(dimensions)])
    for i in range(num_intermediate_samples):
        header.extend([f"P{i}_{j}" for j in range(dimensions)])
    header.extend([f"B_{i}" for i in range(dimensions)])

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for trajectory in trajectories_points:
            writer.writerow(trajectory)

    return trajectories_points

def create_fpc2_intermediate_points(couples, num_intermediate_samples, filename, dimensions=2):
    alpha_values = [0.01 * i for i in range(1, num_intermediate_samples)] + [0.99]

    trajectories_points = []
    for couple in couples:
        a = np.array(couple[:dimensions])
        b = np.array(couple[dimensions:2*dimensions])

        trajectory_points = a.tolist()

        for i in range(num_intermediate_samples):
            distance_from_a = (b-a) * alpha_values[i]
            intermediate_point = a + distance_from_a
            trajectory_points.extend(intermediate_point.tolist())

        trajectory_points.extend(b.tolist())
        trajectories_points.append(trajectory_points)

    # Generate header dynamically
    header = []
    header.extend([f"A_{i}" for i in range(dimensions)])
    for i in range(num_intermediate_samples):
        header.extend([f"P{i}_{j}" for j in range(dimensions)])
    header.extend([f"B_{i}" for i in range(dimensions)])

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for trajectory in trajectories_points:
            writer.writerow(trajectory)

    return trajectories_points

def create_fpc3_intermediate_points(couples, num_intermediate_samples, filename, dimensions=2):
    def normal_vector(v):
        # Generate a random vector of the same dimension as v
        r = np.random.randn(len(v))

        # Substract the projection of r onto v
        projection = np.dot(r, v) / np.dot(v, v) * v
        normal = r - projection

        # Normalize the result
        normal = normal / np.linalg.norm(normal)

        return normal

    trajectories_points = []
    for couple in couples:
        a = np.array(couple[:dimensions])
        b = np.array(couple[dimensions:2*dimensions])

        trajectory_points = a.tolist()

        # Create a point equidistant from a and b but not on the segment [AB]
        distance_from_segment_middle = 2 * np.random.rand() * np.linalg.norm(a - b) + np.linalg.norm(a - b) * 0.1 # Distance in [0.1*AB, 2*AB[
        m = a + (b-a) / 2 + normal_vector(b - a) * distance_from_segment_middle # Chasles Relation

        alpha_values = np.linspace(0.0, 1.0, num_intermediate_samples//2 + 2)[1:-1]
        for alpha in alpha_values:
            vector_from_a = (m-a) * alpha
            intermediate_point = a + vector_from_a
            trajectory_points.extend(intermediate_point.tolist())

        trajectory_points.extend(m.tolist())

        for alpha in alpha_values:
            vector_to_b = (b-m) * alpha
            intermediate_point = m + vector_to_b
            trajectory_points.extend(intermediate_point.tolist())

        trajectory_points.extend(b.tolist())
        trajectories_points.append(trajectory_points)

    # Generate header dynamically
    header = []
    header.extend([f"A_{i}" for i in range(dimensions)])
    for i in range(num_intermediate_samples):
        header.extend([f"P{i}_{j}" for j in range(dimensions)])
    header.extend([f"B_{i}" for i in range(dimensions)])

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for trajectory in trajectories_points:
            writer.writerow(trajectory)

    return trajectories_points

def create_linear_intermediate_points(couples, num_intermediate_samples, filename, dimensions=2):
    trajectories_points = []
    for couple in couples:
        a = np.array(couple[:dimensions])
        b = np.array(couple[dimensions:2*dimensions])
        trajectory_points = []
        trajectory_points.extend(a.tolist())
        # Create intermediate points on the line segment
        for i in range(1, num_intermediate_samples+1):
            alpha = i / (num_intermediate_samples+1)
            intermediate_point = a + alpha * (b - a)
            trajectory_points.extend(intermediate_point.tolist())
            
        trajectory_points.extend(b.tolist())
        
        trajectories_points.append(trajectory_points)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            f"A_{i}" for i in range(dimensions)
        ]
        header.extend([
            f"P{i}_{j}" for i in range(num_intermediate_samples) for j in range(dimensions)
        ])
        header.extend([
            f"B_{i}" for i in range(dimensions)
        ])
        writer.writerow(header)
        for trajectory in trajectories_points:
            writer.writerow(trajectory)
    
    return trajectories_points

def create_random_intermediate_points(couples, num_intermediate_samples, filename, dimensions=2):
    trajectories_points = []
    for couple in couples:
        a = np.array(couple[:dimensions])
        b = np.array(couple[dimensions:2*dimensions])
        trajectory_points = []
        trajectory_points.extend(a.tolist())
        # Create random intermediate points in the whole space
        for _ in range(num_intermediate_samples):
            random_point = np.random.rand(dimensions) * 10  # Random point in the space
            trajectory_points.extend(random_point.tolist())
            
        trajectory_points.extend(b.tolist())
        
        trajectories_points.append(trajectory_points)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            f"A_{i}" for i in range(dimensions)
        ]
        header.extend([
            f"P{i}_{j}" for i in range(num_intermediate_samples) for j in range(dimensions)
        ])
        header.extend([
            f"B_{i}" for i in range(dimensions)
        ])
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
        
    results_dir_fpc1 = "exp_invalidate_metrics_fpc/generated/metrics/fpc1"
    fpc1_metrics_values = get_metrics_values(results_dir_fpc1)
    results_dir_fpc2 = "exp_invalidate_metrics_fpc/generated/metrics/fpc2"
    fpc2_metrics_values = get_metrics_values(results_dir_fpc2)
    results_dir_fpc3 = "exp_invalidate_metrics_fpc/generated/metrics/fpc3"
    fpc3_metrics_values = get_metrics_values(results_dir_fpc3)
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
        header = ["Metric", "Ideal Value", "Random Value", "FPC 1 Value", "FPC 2 Value", "FPC 3 Value"]
        writer.writerow(header)

        # Write rows: models as rows, (k, p) as columns, mean+-std as values
        row = [
            "LCS",
            f"{linear_metrics_values['LCS'][0]:.2f} +- {linear_metrics_values['LCS'][1]:.2f}",
            f"{random_metrics_values['LCS'][0]:.2f} +- {random_metrics_values['LCS'][1]:.2f}",
            f"{fpc1_metrics_values['LCS'][0]:.2f} +- {fpc1_metrics_values['LCS'][1]:.2f}",
            f"{fpc2_metrics_values['LCS'][0]:.2f} +- {fpc2_metrics_values['LCS'][1]:.2f}",
            f"{fpc3_metrics_values['LCS'][0]:.2f} +- {fpc3_metrics_values['LCS'][1]:.2f}",
        ]
        writer.writerow(row)
        row = [
            "Correspondence",
            f"{linear_metrics_values['Correspondence'][0]:.2f} +- {linear_metrics_values['Correspondence'][1]:.2f}",
            f"{random_metrics_values['Correspondence'][0]:.2f} +- {random_metrics_values['Correspondence'][1]:.2f}",
            f"{fpc1_metrics_values['Correspondence'][0]:.2f} +- {fpc1_metrics_values['Correspondence'][1]:.2f}",
            f"{fpc2_metrics_values['Correspondence'][0]:.2f} +- {fpc2_metrics_values['Correspondence'][1]:.2f}",
            f"{fpc3_metrics_values['Correspondence'][0]:.2f} +- {fpc3_metrics_values['Correspondence'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            f"{linear_metrics_values['Smoothness CLAP'][0]:.2f} +- {linear_metrics_values['Smoothness CLAP'][1]:.2f}",
            f"{random_metrics_values['Smoothness CLAP'][0]:.2f} +- {random_metrics_values['Smoothness CLAP'][1]:.2f}",
            f"{fpc1_metrics_values['Smoothness CLAP'][0]:.2f} +- {fpc1_metrics_values['Smoothness CLAP'][1]:.2f}",
            f"{fpc2_metrics_values['Smoothness CLAP'][0]:.2f} +- {fpc2_metrics_values['Smoothness CLAP'][1]:.2f}",
            f"{fpc3_metrics_values['Smoothness CLAP'][0]:.2f} +- {fpc3_metrics_values['Smoothness CLAP'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            f"{linear_metrics_values['Intermediateness'][0]:.2f} +- {linear_metrics_values['Intermediateness'][1]:.2f}",
            f"{random_metrics_values['Intermediateness'][0]:.2f} +- {random_metrics_values['Intermediateness'][1]:.2f}",
            f"{fpc1_metrics_values['Intermediateness'][0]:.2f} +- {fpc1_metrics_values['Intermediateness'][1]:.2f}",
            f"{fpc2_metrics_values['Intermediateness'][0]:.2f} +- {fpc2_metrics_values['Intermediateness'][1]:.2f}",
            f"{fpc3_metrics_values['Intermediateness'][0]:.2f} +- {fpc3_metrics_values['Intermediateness'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CDPAM",
            f"{linear_metrics_values['Smoothness CDPAM'][0]:.2f} +- {linear_metrics_values['Smoothness CDPAM'][1]:.2f}",
            f"{random_metrics_values['Smoothness CDPAM'][0]:.2f} +- {random_metrics_values['Smoothness CDPAM'][1]:.2f}",
            f"{fpc1_metrics_values['Smoothness CDPAM'][0]:.2f} +- {fpc1_metrics_values['Smoothness CDPAM'][1]:.2f}",
            f"{fpc2_metrics_values['Smoothness CDPAM'][0]:.2f} +- {fpc2_metrics_values['Smoothness CDPAM'][1]:.2f}",
            f"{fpc3_metrics_values['Smoothness CDPAM'][0]:.2f} +- {fpc3_metrics_values['Smoothness CDPAM'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (0, 2)",
            f"{linear_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{fpc1_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {fpc1_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{fpc2_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {fpc2_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{fpc3_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {fpc3_metrics_values['Sobolev (0, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (1, 2)",
            f"{linear_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{fpc1_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {fpc1_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{fpc2_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {fpc2_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{fpc3_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {fpc3_metrics_values['Sobolev (1, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
    