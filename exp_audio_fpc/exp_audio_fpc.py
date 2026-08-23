import csv
import os
import numpy as np
import torch

from compute_embeddings import compute_trajectories_embeddings
from synthesize_audios import synthesize_audios_trajectories

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
np.random.seed(seed)
torch.manual_seed(seed)

num_intermediate_samples = 11

# --------------------------------------------------------
#               Compute random trajectories              -
# --------------------------------------------------------

print("Computing random trajectories...")

## Load trajectories
trajectories_path = "exp_embeddings_linearity/generated/random_thetas_trajectories.csv"
print(f"Loading random trajectories from {trajectories_path}")
trajectories = load_trajectories_from_csv(trajectories_path)

## Compute metrics
results_dir = f"exp_audio_fpc/generated/results/random/"
os.makedirs(results_dir, exist_ok=True)
model_name = "MERT_v1-330M"
embeddings_dir = f"exp_embeddings_linearity/generated/random_embeddings/{model_name}"

compute_metrics = False
if compute_metrics:
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    audios_dir = f"exp_embeddings_linearity/generated/random_audio"
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_embeddings_linearity/generated/random_embeddings/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    audios_dir = f"exp_embeddings_linearity/generated/random_audio"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audios_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

# --------------------------------------------------------
#          Compute experiments Ideal points              -
# --------------------------------------------------------

print("Computing ideal trajectories...")

## Load trajectories
trajectories_path = "exp_embeddings_linearity/generated/trajectories.csv"
print(f"Loading trajectories from {trajectories_path}")
trajectories = load_trajectories_from_csv(trajectories_path)

## Compute metrics
results_dir = f"exp_audio_fpc/generated/results/experiment/"
os.makedirs(results_dir, exist_ok=True)

compute_metrics = False
if compute_metrics:
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_embeddings_linearity/generated/embeddings/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    audios_dir = f"exp_embeddings_linearity/generated/audio"
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_embeddings_linearity/generated/embeddings/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    audios_dir = f"exp_embeddings_linearity/generated/audio"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audios_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audios_dir)

# --------------------------------------------------------
#          Compute experiments FPNUC points              -
# --------------------------------------------------------

print("Computing fpnuc trajectories...")

from compute_fpc_audios import create_fpnuc_intermediate_points

fpnuc_process = False
if fpnuc_process:
    # Compute parameters trajectories
    trajectories_filepath = create_fpnuc_intermediate_points(num_intermediate_samples=num_intermediate_samples, dirname="exp_audio_fpc/generated/trajectories/")
    print(f"Loading fpnuc trajectories from {trajectories_filepath}")
    trajectories = load_trajectories_from_csv(trajectories_filepath)

    ## Generate audios
    audio_dir = "exp_audio_fpc/generated/audios/fpnuc"
    synthesize_audios_trajectories(trajectories, logscale = True, audio_dir=audio_dir)

    ## Compute embeddings
    models = ["LaionCLAP_audio", "MERT_v1-330M"]
    embeddings_dir = "exp_audio_fpc/generated/embeddings/fpnuc"
    compute_trajectories_embeddings(models, trajectories, audio_dir=audio_dir, embeddings_dir=embeddings_dir)

compute_metrics = False
if compute_metrics:
    results_dir = f"exp_audio_fpc/generated/results/fpnuc/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/fpnuc/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/fpnuc/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

# --------------------------------------------------------
#          Compute experiments normalized FPCC points              -
# --------------------------------------------------------

print("Computing normalized fpcc trajectories...")
normalized_fpcc_process = True
if normalized_fpcc_process:
    from compute_fpc_audios import create_normalized_fpcc_intermediate_points

    # Compute parameters trajectories
    trajectories_filepath = create_normalized_fpcc_intermediate_points(num_intermediate_samples=num_intermediate_samples, dirname="exp_audio_fpc/generated/trajectories/")
    print(f"Loading normalized fpcc trajectories from {trajectories_filepath}")
    trajectories = load_trajectories_from_csv(trajectories_filepath)

    ## Generate audios
    audio_dir = "exp_audio_fpc/generated/audios/normalized_fpcc"
    synthesize_audios_trajectories(trajectories, logscale = True, audio_dir=audio_dir)

    ## Compute embeddings
    models = ["LaionCLAP_audio", "MERT_v1-330M"]
    embeddings_dir = "exp_audio_fpc/generated/embeddings/normalized_fpcc/"
    compute_trajectories_embeddings(models, trajectories, audio_dir=audio_dir, embeddings_dir=embeddings_dir)

compute_metrics = True
if compute_metrics:
    results_dir = f"exp_audio_fpc/generated/results/normalized_fpcc/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/normalized_fpcc/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/normalized_fpcc/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

# --------------------------------------------------------
#          Compute experiments FPCC points              -
# --------------------------------------------------------

fpcc_process = False
if fpcc_process:
    print("Computing fpcc trajectories...")

    from compute_fpc_audios import create_fpcc_intermediate_points

    # Compute parameters trajectories
    trajectories_filepath = create_fpcc_intermediate_points(num_intermediate_samples=num_intermediate_samples, dirname="exp_audio_fpc/generated/trajectories/")
    print(f"Loading fpcc trajectories from {trajectories_filepath}")
    trajectories = load_trajectories_from_csv(trajectories_filepath)

    ## Generate audios
    audio_dir = "exp_audio_fpc/generated/audios/fpcc"
    synthesize_audios_trajectories(trajectories, logscale = True, audio_dir=audio_dir)

    ## Compute embeddings
    models = ["LaionCLAP_audio", "MERT_v1-330M"]
    embeddings_dir = "exp_audio_fpc/generated/embeddings/fpcc/"
    compute_trajectories_embeddings(models, trajectories, audio_dir=audio_dir, embeddings_dir=embeddings_dir)

compute_metrics = False
if compute_metrics:
    results_dir = f"exp_audio_fpc/generated/results/fpcc/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/fpcc/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/fpcc/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

# ----------------------------------------
#                Make table              -
# ----------------------------------------

make_table("exp_audio_fpc/generated/results")