import csv
import os

from sobolev_distance import compute_sobolev_distances
from morphfader_smoothness_clap_corr import compute_smoothness_clap_corr
from soundmorpher_correspondence_mfccs import compute_soundmorpher_correspondence_mfccs
from compute_cdpam import compute_cdpam
from soundmorpher_intermediateness_total_cdpam import compute_intermediateness_total_cdpam
from soundmorpher_smoothness_cdpam import compute_smoothness_mean_cdpam
from make_table import make_table

def load_trajectories_from_csv(filename):
    trajectories = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for row in reader:
            # Convert each value to float and group into tuples of 5 parameters
            points = [
                list(map(float, row[i:i+5]))
                for i in range(0, len(row), 5)
            ]
            trajectories.append(points)

    return trajectories

seed = 42

num_intermediate_samples = 11

# --------------------------------------------------------
#               Compute random trajectories              -
# --------------------------------------------------------

## Load trajectories
trajectories = load_trajectories_from_csv("exp_embeddings_linearity/generated/random_thetas_trajectories.csv")

## Compute metrics
results_dir = f"exp_audio_fpc/generated/results/random/"
os.makedirs(results_dir, exist_ok=True)
model_name = "MERT_v1-330M"
embeddings_dir = f"exp_embeddings_linearity/generated/random_embeddings/{model_name}"
# compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
# compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
# compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
# compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
# compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

# model_name = "MFCC"
# audios_dir = f"exp_embeddings_linearity/generated/random_audio"
# compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

# model_name = "LaionCLAP_audio"
# embeddings_dir = f"exp_embeddings_linearity/generated/random_embeddings/{model_name}"
# compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

model_name = "CDPAM"
audios_dir = f"exp_embeddings_linearity/generated/random_audio"
compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audios_dir)
compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)
compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

# --------------------------------------------------------
#                Compute experiments points              -
# --------------------------------------------------------

## Load trajectories
trajectories = load_trajectories_from_csv("exp_embeddings_linearity/generated/trajectories.csv")

## Compute metrics
results_dir = f"exp_audio_fpc/generated/results/experiment/"
os.makedirs(results_dir, exist_ok=True)
model_name = "MERT_v1-330M"
embeddings_dir = f"exp_embeddings_linearity/generated/embeddings/{model_name}"
# compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
# compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
# compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
# compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
# compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

# model_name = "MFCC"
# audios_dir = f"exp_embeddings_linearity/generated/audio"
# compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

# model_name = "LaionCLAP_audio"
# embeddings_dir = f"exp_embeddings_linearity/generated/embeddings/{model_name}"
# compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

model_name = "CDPAM"
audios_dir = f"exp_embeddings_linearity/generated/audio"
compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audios_dir)
compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)
compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

# ----------------------------------------
#                Make table              -
# ----------------------------------------

make_table("exp_audio_fpc/generated/results")