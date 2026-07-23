import csv
import os

from generate_thetas import generate_and_save_couples, generate_and_save_trajectories, load_trajectories_from_csv
from sobolev_distance import compute_sobolev_distances, make_table
from synthesize_audios import synthesize_audios_trajectories
from compute_embeddings import compute_trajectories_embeddings

from random_trajectories import generate_and_save_random_trajectories

seed = 42

number_of_couples = 1000
num_intermediate_samples = 10

# --------------------------------------------------------
#               Compute random trajectories              -
# --------------------------------------------------------

# # 1. Generate random trajectories of points in a 2D space
# generate_and_save_random_trajectories(seed, number_of_couples, num_intermediate_samples, filename="exp_embeddings_linearity/generated/random_thetas_trajectories.csv")

# ## 2. Load the generated trajectories
# trajectories = load_trajectories_from_csv("exp_embeddings_linearity/generated/random_thetas_trajectories.csv")

# ## 3. Generate the audios
# audio_dir = "exp_embeddings_linearity/generated/random_audio"
# os.makedirs(audio_dir, exist_ok=True)
# # synthesize_audios_trajectories(trajectories, logscale=False, audio_dir=audio_dir)

# ## 4. Compute embeddings
# models = ["LaionCLAP_audio", "LaionCLAP_music", "MSCLAP", "MERT_v1-95M", "MERT_v1-330M", "MERT_v0-public", "VGGish"]
# embeddings_dir = "exp_embeddings_linearity/generated/random_embeddings/"
# compute_trajectories_embeddings(models, trajectories, audio_dir, embeddings_dir)

# ## 5. Compute sobolev distance
# results_dir = f"exp_embeddings_linearity/generated/results/random/"
# os.makedirs(results_dir, exist_ok=True)
# for model_name in models:
#     embeddings_dir = f"exp_embeddings_linearity/generated/random_embeddings/{model_name}"
#     compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)

# # --------------------------------------------------------
# #                Compute experiments points              -
# # --------------------------------------------------------

# ## 1. Generate couples of points in a 2D space
# points_couples_filename = generate_and_save_couples(seed, number_of_couples)

# ## 2. Load the generated couples
# trajectories_filename = generate_and_save_trajectories(seed, points_couples_filename, num_intermediate_samples, filename="exp_embeddings_linearity/generated/trajectories.csv")
# trajectories = load_trajectories_from_csv(trajectories_filename)

# ## 3. Generate the audios
# audio_dir = "exp_embeddings_linearity/generated/audio"
# os.makedirs(audio_dir, exist_ok=True)

# synthesize_audios_trajectories(trajectories, logscale=False, audio_dir=audio_dir)

# ## 4. Compute embeddings
# models = ["LaionCLAP_audio", "LaionCLAP_music", "MSCLAP", "MERT_v1-95M", "MERT_v1-330M", "MERT_v0-public", "VGGish"]

# embeddings_dir = "exp_embeddings_linearity/generated/embeddings/"
# compute_trajectories_embeddings(models, trajectories, audio_dir, embeddings_dir)

# ## 5. Compute sobolev distance
# results_dir = f"exp_embeddings_linearity/generated/results/experiment/"
# os.makedirs(results_dir, exist_ok=True)
# for model_name in models:
#     embeddings_dir = f"exp_embeddings_linearity/generated/embeddings/{model_name}"
#     compute_sobolev_distances(embeddings_dir, results_dir, model_name, trajectories, num_intermediate_samples)

models = ["VGGish", "MSCLAP", "MERT_v0-public", "MERT_v1-95M", "MERT_v1-330M", "LaionCLAP_audio", "LaionCLAP_music"]
results_dir = f"exp_embeddings_linearity/generated/results/"
make_table(results_dir, models)