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
#          Compute experiments NUC points              -
# --------------------------------------------------------

print("Computing nuc trajectories...")

from compute_fpc_audios import create_nuc_intermediate_points

nuc_process = False
if nuc_process:
    # Compute parameters trajectories
    trajectories_filepath = create_nuc_intermediate_points(num_intermediate_samples=num_intermediate_samples, dirname="exp_audio_fpc/generated/trajectories/")
    print(f"Loading nuc trajectories from {trajectories_filepath}")
    trajectories = load_trajectories_from_csv(trajectories_filepath)

    ## Generate audios
    audio_dir = "exp_audio_fpc/generated/audios/nuc"
    synthesize_audios_trajectories(trajectories, logscale = True, audio_dir=audio_dir)

    ## Compute embeddings
    models = ["LaionCLAP_audio", "MERT_v1-330M"]
    embeddings_dir = "exp_audio_fpc/generated/embeddings/nuc"
    compute_trajectories_embeddings(models, trajectories, audio_dir=audio_dir, embeddings_dir=embeddings_dir)

compute_metrics = False
if compute_metrics:
    results_dir = f"exp_audio_fpc/generated/results/nuc/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/nuc/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/nuc/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

# --------------------------------------------------------
#          Compute experiments normalized EQC points              -
# --------------------------------------------------------

print("Computing normalized eqc trajectories...")
normalized_eqc_process = False
if normalized_eqc_process:
    from compute_fpc_audios import create_normalized_eqc_intermediate_points

    # Compute parameters trajectories
    trajectories_filepath = create_normalized_eqc_intermediate_points(num_intermediate_samples=num_intermediate_samples, dirname="exp_audio_fpc/generated/trajectories/")
    print(f"Loading normalized eqc trajectories from {trajectories_filepath}")
    trajectories = load_trajectories_from_csv("exp_audio_fpc/generated/trajectories/normalized_eqc_trajectories.csv")#trajectories_filepath)
    
    ## Generate audios
    audio_dir = "exp_audio_fpc/generated/audios/normalized_eqc"
    synthesize_audios_trajectories(trajectories, logscale = True, audio_dir=audio_dir)

    ## Compute embeddings
    models = ["LaionCLAP_audio", "MERT_v1-330M"]
    embeddings_dir = "exp_audio_fpc/generated/embeddings/normalized_eqc/"
    compute_trajectories_embeddings(models, trajectories, audio_dir=audio_dir, embeddings_dir=embeddings_dir)

compute_metrics = False
if compute_metrics:
    results_dir = f"exp_audio_fpc/generated/results/normalized_eqc/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/normalized_eqc/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/normalized_eqc/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

# --------------------------------------------------------
#          Compute experiments EQC points              -
# --------------------------------------------------------

eqc_process = False
if eqc_process:
    print("Computing eqc trajectories...")

    from compute_fpc_audios import create_eqc_intermediate_points

    # Compute parameters trajectories
    trajectories_filepath = create_eqc_intermediate_points(num_intermediate_samples=num_intermediate_samples, dirname="exp_audio_fpc/generated/trajectories/")
    print(f"Loading eqc trajectories from {trajectories_filepath}")
    trajectories = load_trajectories_from_csv(trajectories_filepath)

    ## Generate audios
    audio_dir = "exp_audio_fpc/generated/audios/eqc"
    synthesize_audios_trajectories(trajectories, logscale = True, audio_dir=audio_dir)

    ## Compute embeddings
    models = ["LaionCLAP_audio", "MERT_v1-330M"]
    embeddings_dir = "exp_audio_fpc/generated/embeddings/eqc/"
    compute_trajectories_embeddings(models, trajectories, audio_dir=audio_dir, embeddings_dir=embeddings_dir)

compute_metrics = False
if compute_metrics:
    results_dir = f"exp_audio_fpc/generated/results/eqc/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/eqc/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    model_name = "MFCC"
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/eqc/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    model_name = "CDPAM"
    compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

# -----------------------------------------------------------------------------
#          Compute random set of parameters to synthesize and encode          -
# -----------------------------------------------------------------------------
from compute_fpc_audios import generate_random_points
from synthesize_audios import synthesize_audios_points
from compute_embeddings import compute_points_embeddings

compute_random_audios= True
if compute_random_audios:
    points_filename = "exp_audio_fpc/generated/random_parameters_points.csv"
    generate_random_points(num_points=15000, points_filename=points_filename, seed=seed)

    audio_dir = "exp_audio_fpc/generated/audios/random_audios"
    synthesize_audios_points(points_filename, logscale = True, audio_dir=audio_dir)

    random_embeddings_dir = "exp_audio_fpc/generated/embeddings/random_embeddings"
    compute_points_embeddings(["LaionCLAP_audio", "MERT_v1-330M"], audio_dir, random_embeddings_dir)

# ------------------------------------------------------------------
#          Get embeddings EQC pointsn by random sampling          -
# ------------------------------------------------------------------

embeddings_eqc_process = True
if embeddings_eqc_process:
    print("Computing embeddings EQC trajectories...")

    from compute_fpc_audios import get_embeddings_eqc_intermediate_points

    for model_name in ["LaionCLAP_audio", "MERT_v1-330M"]:
        # Compute parameters trajectories
        trajectories_filepath = get_embeddings_eqc_intermediate_points(embedding_model=model_name, num_intermediate_samples=num_intermediate_samples, random_points_embeddings_dir=random_embeddings_dir, trajectories_embeddings_dir=f"exp_audio_fpc/generated/embeddings/emb_eqc", results_dir=f"exp_audio_fpc/generated/results/emb_eqc")
        print(f"Loading eqc trajectories from {trajectories_filepath}")

compute_metrics = True
if compute_metrics:
    results_dir = f"exp_audio_fpc/generated/results/emb_eqc/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/emb_eqc/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    # model_name = "MFCC"
    # compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/emb_eqc/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    # model_name = "CDPAM"
    # compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    # compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    # compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)


# ------------------------------------------------------------------
#          Get embeddings NUC pointsn by random sampling          -
# ------------------------------------------------------------------

embeddings_nuc_process = True
if embeddings_nuc_process:
    print("Computing embeddings NUC trajectories...")

    from compute_fpc_audios import get_embeddings_nuc_intermediate_points

    for model_name in ["LaionCLAP_audio", "MERT_v1-330M"]:
        # Compute parameters trajectories
        trajectories_filepath = get_embeddings_nuc_intermediate_points(embedding_model=model_name, num_intermediate_samples=num_intermediate_samples, random_points_embeddings_dir=f"exp_embeddings_linearity/generated/random_embeddings/", trajectories_embeddings_dir=f"exp_audio_fpc/generated/embeddings/emb_nuc", results_dir=f"exp_audio_fpc/generated/results/emb_nuc")
        print(f"Loading nuc trajectories from {trajectories_filepath}")

compute_metrics = True
if compute_metrics:
    results_dir = f"exp_audio_fpc/generated/results/emb_nuc/"
    model_name = "MERT_v1-330M"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/emb_nuc/{model_name}"
    compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)
    compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)
    compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=embeddings_dir)

    # model_name = "MFCC"
    # compute_soundmorpher_correspondence_mfccs(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

    model_name = "LaionCLAP_audio"
    embeddings_dir = f"exp_audio_fpc/generated/embeddings/emb_nuc/{model_name}"
    compute_smoothness_clap_corr(results_dir, model_name, trajectories, embeddings_folder=embeddings_dir)

    # model_name = "CDPAM"
    # compute_cdpam(results_dir, trajectories, audios_or_embeddings_folder=audio_dir)
    # compute_intermediateness_total_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)
    # compute_smoothness_mean_cdpam(results_dir, model_name, trajectories, audios_or_embeddings_folder=audio_dir)

# ----------------------------------------
#                Make table              -
# ----------------------------------------

make_table("exp_audio_fpc/generated/results")