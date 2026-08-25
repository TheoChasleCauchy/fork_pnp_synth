import os
import torch
from tqdm import tqdm
import csv

embeddings = {
    "clap": [1, 512],
    "mert": [1, 1024],
    "cdpam": [1, 512],
    "mfcc": [20, 256] # For a 5 second audio at 44100Hz
}

def sample_couples_specific_dimensions_space(n_couples, dirname):
    os.makedirs(dirname, exist_ok=True)

    for embedding, dimensions in embeddings.items():
        if len(dimensions) == 1:
            # Vector case (1D)
            dim = dimensions[0]
            couples = torch.rand(n_couples, 2, dim) * 10
            filename = f"{dirname}/{embedding}_couples.pt"
            torch.save(couples, filename)
        else:
            # Matrix case (2D)
            rows, cols = dimensions
            couples = torch.rand(n_couples, 2, rows, cols) * 10
            filename = f"{dirname}/{embedding}_couples.pt"
            torch.save(couples, filename)

def create_eqc_intermediate_points(num_intermediate_samples, dirname):
    for embedding in tqdm(embeddings.keys(), desc="Processing embeddings"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_couples.pt")

        trajectories = []

        for couple in tqdm(couples, desc=f"Processing couples for embedding {embedding}"):
            a, b = couple[0], couple[1]

            # Initialize trajectory with point A
            trajectory = [a]

            # Generate intermediate points between A and B
            for i in range(1, num_intermediate_samples + 1):
                # Random direction in the same shape as A
                random_direction = torch.rand_like(a)
                random_direction = random_direction / torch.linalg.norm(random_direction)  # Normalize

                # Distance from A proportional to progress
                distance_from_a = torch.linalg.norm(a - b) * i / (num_intermediate_samples + 1)
                intermediate_point = a + distance_from_a * random_direction
                trajectory.append(intermediate_point)

            # Add point B to complete the trajectory
            trajectory.append(b)
            assert len(trajectory) == num_intermediate_samples + 2
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_eqc_trajectories.pt"
        torch.save(trajectories_tensor, filename)

def create_nuc_intermediate_points(num_intermediate_samples, dirname):
    alpha_values = [0.01 * i for i in range(1, num_intermediate_samples)] + [0.99]

    for embedding in tqdm(embeddings.keys(), desc="Processing embeddings"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_couples.pt")

        trajectories = []
        for couple in tqdm(couples, desc=f"Processing couples for embedding {embedding}"):
            a, b = couple[0], couple[1]

            # Initialize trajectory with point A
            trajectory = [a]

            for i in range(num_intermediate_samples):
                distance_from_a = (b-a) * alpha_values[i]
                intermediate_point = a + distance_from_a
                trajectory.append(intermediate_point)

            trajectory.append(b)
            assert len(trajectory) == num_intermediate_samples + 2
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_nuc_trajectories.pt"
        torch.save(trajectories_tensor, filename)

def create_tc_intermediate_points(num_intermediate_samples, dirname):
    def normal_vector(v):
        # Generate a random tensor of the same shape as v
        r = torch.rand_like(v)

        # Flatten for dot product calculations
        r_flat = r.reshape(-1)
        v_flat = v.reshape(-1)

        # Compute projection and normal vector
        projection = torch.dot(r_flat, v_flat) / torch.dot(v_flat, v_flat) * v
        normal = r - projection

        # Normalize the result
        normal = normal / torch.linalg.norm(normal)

        return normal

    for embedding in tqdm(embeddings.keys(), desc="Processing embeddings"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_couples.pt")

        trajectories = []
        for couple in tqdm(couples, desc=f"Processing couples for embedding {embedding}"):
            a, b = couple[0], couple[1]

            trajectory = [a]

            # Create a point equidistant from a and b but not on the segment [AB]
            distance_from_segment_middle = 2 * torch.rand(1, device=a.device) * torch.linalg.norm(a - b) + torch.linalg.norm(a - b) * 0.1 # Distance in [0.1*AB, 2*AB[
            m = a + (b-a) / 2 + normal_vector(b - a) * distance_from_segment_middle # Chasles Relation

            alpha_values = torch.linspace(0.0, 1.0, num_intermediate_samples//2 + 2)[1:-1]
            for alpha in alpha_values:
                vector_from_a = (m-a) * alpha
                intermediate_point = a + vector_from_a
                trajectory.append(intermediate_point)

            trajectory.append(m)

            for alpha in alpha_values:
                vector_to_b = (b-m) * alpha
                intermediate_point = m + vector_to_b
                trajectory.append(intermediate_point)

            trajectory.append(b)
            assert len(trajectory) == num_intermediate_samples + 2
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_tc_trajectories.pt"
        torch.save(trajectories_tensor, filename)

def create_linear_intermediate_points(num_intermediate_samples, dirname):

    for embedding in tqdm(embeddings.keys(), desc="Processing embeddings"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_couples.pt")

        trajectories = []
        for couple in tqdm(couples, desc=f"Processing couples for embedding {embedding}"):
            a, b = couple[0], couple[1]

            trajectory = [a]
            
            # Create intermediate points on the line segment
            for i in range(1, num_intermediate_samples+1):
                alpha = i / (num_intermediate_samples+1)
                intermediate_point = a + alpha * (b - a)
                trajectory.append(intermediate_point)
                
            trajectory.append(b)
            assert len(trajectory) == num_intermediate_samples + 2
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_linear_trajectories.pt"
        torch.save(trajectories_tensor, filename)

def create_random_intermediate_points(num_intermediate_samples, dirname):
    for embedding in tqdm(embeddings.keys(), desc="Processing embeddings"):
        # Load couples as torch tensor
        couples = torch.load(f"{dirname}/{embedding}_couples.pt")
        
        trajectories = []
        for couple in tqdm(couples, desc=f"Processing couples for embedding {embedding}"):
            a, b = couple[0], couple[1]

            trajectory = [a]

            # Create random intermediate points in the whole space
            for _ in range(num_intermediate_samples):
                random_point = torch.rand_like(a) * 10  # Random point in the space
                trajectory.append(random_point)
            
            trajectory.append(b)
            trajectories.append(torch.stack(trajectory))

        # Stack all trajectories into a single tensor (shape: [n_trajectories, num_intermediate_samples + 2, *dimensions])
        trajectories_tensor = torch.stack(trajectories)

        # Save as PyTorch tensor
        filename = f"{dirname}/{embedding}_random_trajectories.pt"
        torch.save(trajectories_tensor, filename)

# Make a table of the mean result of each metric
def make_table():
    import re
    
    def get_metrics_values(results_dir):
        metrics_values = {}
        
        # Get Smoothness Clap metric value
        clap_smoothness_clap_csv_path = os.path.join(results_dir, "clap_smoothness_clap_corr_values.csv")
        with open(clap_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["CLAP Smoothness"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)

        mert_smoothness_clap_csv_path = os.path.join(results_dir, "mert_smoothness_clap_corr_values.csv")
        with open(mert_smoothness_clap_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_clap = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_clap_corr, std_smoothness_clap_corr = map(float, mean_std_clap)
            metrics_values["MERT Smoothness CLAP"] = (mean_smoothness_clap_corr, std_smoothness_clap_corr)
        
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
        mfcc_correspondence_csv_path = os.path.join(results_dir, "mfcc_soundmorpher_correspondence_mfccs_values.csv")
        with open(mfcc_correspondence_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
            metrics_values["MFCC Correspondence"] = (mean_correspondence, std_correspondence)

        mert_correspondence_csv_path = os.path.join(results_dir, "mert_soundmorpher_correspondence_mfccs_values.csv")
        with open(mert_correspondence_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_correspondence = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_correspondence, std_correspondence = map(float, mean_std_correspondence)
            metrics_values["MERT Correspondence"] = (mean_correspondence, std_correspondence)
        
        # Get Intermediateness value
        cdpam_intermediateness_csv_path = os.path.join(results_dir, "cdpam_intermediateness_total_cdpam_values.csv")
        with open(cdpam_intermediateness_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
            metrics_values["CDPAM Intermediateness"] = (mean_intermediateness, std_intermediateness)

        mert_intermediateness_csv_path = os.path.join(results_dir, "mert_intermediateness_total_cdpam_values.csv")
        with open(mert_intermediateness_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_intermediateness = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_intermediateness, std_intermediateness = map(float, mean_std_intermediateness)
            metrics_values["MERT Intermediateness"] = (mean_intermediateness, std_intermediateness)

        # Get Smoothness CDPAM value
        cdpam_smoothness_cdpam_csv_path = os.path.join(results_dir, "cdpam_smoothness_mean_cdpam_values.csv")
        with open(cdpam_smoothness_cdpam_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
            metrics_values["CDPAM Smoothness CDPAM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)

        mert_smoothness_cdpam_csv_path = os.path.join(results_dir, "mert_smoothness_mean_cdpam_values.csv")
        with open(mert_smoothness_cdpam_csv_path, 'r') as f:
            reader = list(csv.reader(f))
            row = reader[-1] # Get the last row where the mean value is
            value_string = row[1]
            mean_std_smoothness_cdpam = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value_string)
            mean_smoothness_cdpam, std_smoothness_cdpam = map(float, mean_std_smoothness_cdpam)
            metrics_values["MERT Smoothness CDPAM"] = (mean_smoothness_cdpam, std_smoothness_cdpam)
        
        return metrics_values
        
    results_dir_eqc = "exp_invalidate_metrics_fpc/generated/metrics/eqc"
    eqc_metrics_values = get_metrics_values(results_dir_eqc)
    results_dir_nuc = "exp_invalidate_metrics_fpc/generated/metrics/nuc"
    nuc_metrics_values = get_metrics_values(results_dir_nuc)
    results_dir_tc = "exp_invalidate_metrics_fpc/generated/metrics/tc"
    tc_metrics_values = get_metrics_values(results_dir_tc)
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
        header = ["Metric", "Encoder", "Ideal Value", "Random Value", "FPC 1 Value", "FPC 2 Value", "FPC 3 Value"]
        writer.writerow(header)

        # Write rows: models as rows, (k, p) as columns, mean+-std as values
        row = [
            "Correspondence", 
            "MFCC",
            f"{linear_metrics_values['MFCC Correspondence'][0]:.2f} +- {linear_metrics_values['MFCC Correspondence'][1]:.2f}",
            f"{random_metrics_values['MFCC Correspondence'][0]:.2f} +- {random_metrics_values['MFCC Correspondence'][1]:.2f}",
            f"{eqc_metrics_values['MFCC Correspondence'][0]:.2f} +- {eqc_metrics_values['MFCC Correspondence'][1]:.2f}",
            f"{nuc_metrics_values['MFCC Correspondence'][0]:.2f} +- {nuc_metrics_values['MFCC Correspondence'][1]:.2f}",
            f"{tc_metrics_values['MFCC Correspondence'][0]:.2f} +- {tc_metrics_values['MFCC Correspondence'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            "L-CLAP audio",
            f"{linear_metrics_values['CLAP Smoothness'][0]:.2f} +- {linear_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{random_metrics_values['CLAP Smoothness'][0]:.2f} +- {random_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{eqc_metrics_values['CLAP Smoothness'][0]:.2f} +- {eqc_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{nuc_metrics_values['CLAP Smoothness'][0]:.2f} +- {nuc_metrics_values['CLAP Smoothness'][1]:.2f}",
            f"{tc_metrics_values['CLAP Smoothness'][0]:.2f} +- {tc_metrics_values['CLAP Smoothness'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            "CDPAM",
            f"{linear_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {linear_metrics_values['CDPAM Intermediateness'][1]:.2f}",
            f"{random_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {random_metrics_values['CDPAM Intermediateness'][1]:.2f}",
            f"{eqc_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {eqc_metrics_values['CDPAM Intermediateness'][1]:.2f}",
            f"{nuc_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {nuc_metrics_values['CDPAM Intermediateness'][1]:.2f}",
            f"{tc_metrics_values['CDPAM Intermediateness'][0]:.2f} +- {tc_metrics_values['CDPAM Intermediateness'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CDPAM",
            "CDPAM",
            f"{linear_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {linear_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}",
            f"{random_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {random_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}",
            f"{eqc_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {eqc_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}",
            f"{nuc_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {nuc_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}",
            f"{tc_metrics_values['CDPAM Smoothness CDPAM'][0]:.2f} +- {tc_metrics_values['CDPAM Smoothness CDPAM'][1]:.2f}"
        ]

        # MERT
        writer.writerow(row)
        row = [
            "Correspondence", 
            "MERT",
            f"{linear_metrics_values['MERT Correspondence'][0]:.2f} +- {linear_metrics_values['MERT Correspondence'][1]:.2f}",
            f"{random_metrics_values['MERT Correspondence'][0]:.2f} +- {random_metrics_values['MERT Correspondence'][1]:.2f}",
            f"{eqc_metrics_values['MERT Correspondence'][0]:.2f} +- {eqc_metrics_values['MERT Correspondence'][1]:.2f}",
            f"{nuc_metrics_values['MERT Correspondence'][0]:.2f} +- {nuc_metrics_values['MERT Correspondence'][1]:.2f}",
            f"{tc_metrics_values['MERT Correspondence'][0]:.2f} +- {tc_metrics_values['MERT Correspondence'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CLAP",
            "MERT",
            f"{linear_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {linear_metrics_values['MERT Smoothness CLAP'][1]:.2f}",
            f"{random_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {random_metrics_values['MERT Smoothness CLAP'][1]:.2f}",
            f"{eqc_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {eqc_metrics_values['MERT Smoothness CLAP'][1]:.2f}",
            f"{nuc_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {nuc_metrics_values['MERT Smoothness CLAP'][1]:.2f}",
            f"{tc_metrics_values['MERT Smoothness CLAP'][0]:.2f} +- {tc_metrics_values['MERT Smoothness CLAP'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Intermediateness",
            "MERT",
            f"{linear_metrics_values['MERT Intermediateness'][0]:.2f} +- {linear_metrics_values['MERT Intermediateness'][1]:.2f}",
            f"{random_metrics_values['MERT Intermediateness'][0]:.2f} +- {random_metrics_values['MERT Intermediateness'][1]:.2f}",
            f"{eqc_metrics_values['MERT Intermediateness'][0]:.2f} +- {eqc_metrics_values['MERT Intermediateness'][1]:.2f}",
            f"{nuc_metrics_values['MERT Intermediateness'][0]:.2f} +- {nuc_metrics_values['MERT Intermediateness'][1]:.2f}",
            f"{tc_metrics_values['MERT Intermediateness'][0]:.2f} +- {tc_metrics_values['MERT Intermediateness'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Smoothness CDPAM",
            "MERT",
            f"{linear_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {linear_metrics_values['MERT Smoothness CDPAM'][1]:.2f}",
            f"{random_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {random_metrics_values['MERT Smoothness CDPAM'][1]:.2f}",
            f"{eqc_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {eqc_metrics_values['MERT Smoothness CDPAM'][1]:.2f}",
            f"{nuc_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {nuc_metrics_values['MERT Smoothness CDPAM'][1]:.2f}",
            f"{tc_metrics_values['MERT Smoothness CDPAM'][0]:.2f} +- {tc_metrics_values['MERT Smoothness CDPAM'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (0, 2)",
            "MERT",
            f"{linear_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{eqc_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {eqc_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{nuc_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {nuc_metrics_values['Sobolev (0, 2)'][1]:.2f}",
            f"{tc_metrics_values['Sobolev (0, 2)'][0]:.2f} +- {tc_metrics_values['Sobolev (0, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
        row = [
            "Sobolev (1, 2)",
            "MERT",
            f"{linear_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {linear_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{random_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {random_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{eqc_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {eqc_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{nuc_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {nuc_metrics_values['Sobolev (1, 2)'][1]:.2f}",
            f"{tc_metrics_values['Sobolev (1, 2)'][0]:.2f} +- {tc_metrics_values['Sobolev (1, 2)'][1]:.2f}"
        ]
        writer.writerow(row)
    