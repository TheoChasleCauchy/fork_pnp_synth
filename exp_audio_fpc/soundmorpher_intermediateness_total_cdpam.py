import torch
import numpy as np
import csv, os
from tqdm import tqdm

import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def mert_process_csv(trajectories: list[np.ndarray], audios_or_embeddings_folder: str, model_name: str):

    from cdpam import lossnet_dfl
    cdpam_loss = lossnet_dfl(1024).to(device)

    # Compute intermediateness value 
    intermediateness_values = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing Intermediateness Total CDPAM", total=len(trajectories))):
        morph_embeddings = []
        for i_theta in range(len(trajectory)):
            embedding = np.load(os.path.join(audios_or_embeddings_folder, f"embedding_{model_name}_row_{i_traj}_AB_I{i_theta}.npy"))
            embedding = torch.from_numpy(embedding).float()
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            morph_embeddings.append(embedding)

        CDPAM_values = []
        for i in range(len(morph_embeddings)-1):
            a1 = morph_embeddings[i].to(device)
            a1 = F.normalize(a1, dim=1)
            a2 = morph_embeddings[i+1].to(device)
            a2 = F.normalize(a2, dim=1)
            CDPAM_value = cdpam_loss.forward(a1,a2)
            CDPAM_values.append(CDPAM_value)

        # intermediateness_value = torch.sum(torch.stack([torch.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]))
        intermediateness_value = torch.stack(CDPAM_values).sum()
        intermediateness_values.append(intermediateness_value.item())

    return intermediateness_values

def cdpam_process_csv(trajectories: list[np.ndarray], audios_folder: str):

    from cdpam import CDPAM
    import librosa
    
    CDPAM_model = CDPAM(dev=device)

    def _load_audio(audio_path):
        """Helper to load audio file using librosa."""
        return librosa.load(audio_path, sr=22050, mono=True)[0]

    # Compute intermediateness value 
    intermediateness_values = []
    for i_traj, trajectory in enumerate(tqdm(trajectories, desc="Computing Intermediateness Total CDPAM", total=len(trajectories))):
        morph_audio_names = []
        for i_theta in range(len(trajectory)):
            # Load the audio files for each theta value
            morph_name = os.path.join(audios_folder, f"audio_row_{i_traj}_AB_I{i_theta}.wav")
            morph_audio_names.append(morph_name)

        morphed_audios = []
        for i_theta in range(len(trajectory)):
            audio = torch.from_numpy(_load_audio(morph_audio_names[i_theta])).float().unsqueeze(0)
            morphed_audios.append(audio)
        

        CDPAM_values = []
        for i in range(len(morphed_audios)-1):
            with torch.no_grad():
                CDPAM_value = CDPAM_model.forward(morphed_audios[i], morphed_audios[i+1]).item()
                CDPAM_values.append(CDPAM_value)
                del CDPAM_value

        # intermediateness_value = torch.sum(torch.stack([torch.linalg.norm(trajectory[i+1] - trajectory[i]) for i in range(len(trajectory)-1)]))
        intermediateness_value = torch.tensor(CDPAM_values).sum()
        intermediateness_values.append(intermediateness_value.item())

    return intermediateness_values

def compute_intermediateness_total_cdpam(results_dir: str, model_name: str, trajectories: list[np.ndarray], audios_or_embeddings_folder: str):
    assert model_name in ["MERT_v1-330M", "CDPAM"]

    match model_name:
        case "MERT_v1-330M":
            intermediateness_values = mert_process_csv(trajectories, audios_or_embeddings_folder, model_name)
        case "CDPAM":
            intermediateness_values = cdpam_process_csv(trajectories, audios_or_embeddings_folder)

    # Write intermediateness values in a csv file
    os.makedirs(os.path.join(results_dir, model_name), exist_ok=True)
    with open(os.path.join(results_dir, model_name, f"{model_name}_intermediateness_total_cdpam_values.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Row", "Intermediateness Total CDPAM"])
        for i, intermediateness_value in enumerate(intermediateness_values):
            writer.writerow([i, intermediateness_value])
        writer.writerow(["Mean Intermediateness", f"{np.mean(intermediateness_values)} +- {np.std(intermediateness_values)}"])