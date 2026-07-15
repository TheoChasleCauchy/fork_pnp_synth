import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

morphing_names = ["metal_to_wood", "range_omega", "range_tau", "range_logp", "range_logD", "range_alpha"]

for morphing_name in morphing_names:
    
    # Configuration
    folder_path = f'generations/{morphing_name}/embeddings'  # Replace with your folder path
    file_pattern = r'value_(.*).npy'

    # Load embeddings and extract values
    embeddings = []
    values = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            match = re.search(file_pattern, filename)
            if match:
                value = match.group(1)
                filepath = os.path.join(folder_path, filename)
                embedding = np.load(filepath)
                embeddings.append(embedding)
                values.append(value)

    # Convert to numpy array
    embeddings_array = np.array(embeddings)

    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_array)

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

    # Annotate each point with its value
    for i, (x, y) in enumerate(embeddings_2d):
        plt.annotate(values[i], (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

    plt.title('2D PCA Visualization of Embeddings')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'generations/{morphing_name}/{morphing_name}_embeddings_visualization.png')
    print("Done !")