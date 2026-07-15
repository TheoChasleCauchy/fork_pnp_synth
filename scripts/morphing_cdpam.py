import argparse
import csv
import re
import torch  # PyTorch for tensor operations
import os  # Library for interacting with the operating system
import cdpam

def compute_cdpam(morphed_audios_names):
    """
    Load the morphed and expected audio files and computes the CDPAM between them.

    Args:
        morphed_audios_names (list[str]): List of paths to morphed audio files.
    
    Returns:
        CDPAM (float): The CDPAM value along the morph.
    """
    morphed_audios = [cdpam.load_audio(audio_name) for audio_name in morphed_audios_names]

    CDPAM = []
    loss_fn = cdpam.CDPAM(dev='cuda:0' if torch.cuda.is_available() else 'cpu', modfolder="/localdata/chasle-t/Work/fork_pnp_synth/models/CDPAM/scratchJNDdefault_best_model.pth")
    for i in range(len(morphed_audios)-1):
        with torch.no_grad():
            loss = loss_fn.forward(morphed_audios[i], morphed_audios[i+1]).item()
        CDPAM.append(loss)
        del loss
    
    return CDPAM

def get_morphing_audios(audios_folder):
    audios = [os.path.join(audios_folder, f) for f in os.listdir(audios_folder) if f.endswith(".wav")]
    return sorted(audios, key=lambda x: int(re.search(r'audio_(\d+)', os.path.basename(x)).group(1)))

def main(morphing_name):
    audios_folder = f"generations/{morphing_name}/audios"

    morphing_audios = get_morphing_audios(audios_folder)
    cdpam_values = compute_cdpam(morphing_audios)

    # Write cdpam values in a csv file
    with open(f"generations/{morphing_name}/{morphing_name}_cdpam_values.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"CDPAM_{i}_to_{i+1}" for i in range(0, len(cdpam_values))])
        writer.writerow(cdpam_values)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load the morphing's audios.")
    parser.add_argument("morphing_name", type=str, help="Name of the morphing")

    # Parse arguments
    args = parser.parse_args()

    # Call main with the directory argument
    main(args.morphing_name)