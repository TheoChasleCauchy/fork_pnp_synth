import numpy as np
import csv

def sample_couples_2D_space(n_couples, filename, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    couples = []
    for _ in range(n_couples):
        a = np.random.rand(2) * 10  # Random point in 2D space
        b = np.random.rand(2) * 10  # Random point in 2D space
        couples.append((a, b))
    
    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "A",
            "B",
        ]
        writer.writerow(header)
        writer.writerows(couples)
    
    return couples

def create_intermediate_points(couples, num_intermediate_samples, filename):
    trajectories_points = []
    for a, b in couples:
        trajectory_points = []
        trajectory_points.append(a)
        # Create intermediate points on the circle
        for i in range(1, num_intermediate_samples+1):
            random_angle = np.random.rand(1) * 2 * np.pi
            distance_from_a = np.linalg.norm(a - b) * i / (num_intermediate_samples+2)
            x = a[0] + distance_from_a * np.cos(random_angle)
            y = a[1] + distance_from_a * np.sin(random_angle)
            trajectory_points.append(np.array([x[0], y[0]]))
            
        trajectory_points.append(b)
        
        trajectories_points.append(trajectory_points)

    # Save to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "A", *[f"P{i}" for i in range(num_intermediate_samples)], "B"
        ]
        writer.writerow(header)
        for trajectory in trajectories_points:
            writer.writerow(trajectory)
    
    return trajectories_points