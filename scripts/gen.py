from pnp_synth.physical import ftm
import soundfile as sf
import os
import numpy as np

audio_dir = "generations/test/audios" # os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)
theta_rows = [[3000, 0.5, 1, -2, i] for i in np.linspace(0.01, 1.0, num=10)]

# wood : [[1500, 6000], [0.015, 0.035], [5, 1000], [-5, -0.5], [0.01, 1.0]]
# metal : [[1500, 8000], [0.15, 1.0], [0.15, 2], [-5, -0.5], [0.1, 1.0]]

for i, row in enumerate(theta_rows):
    # Physical audio synthesis (g). theta -> x
    theta = np.array(row) # icassp23.THETA_COLUMNS
    x = ftm.rectangular_drum(theta, False, **ftm.constants).cpu()
    x = x / max(x)

    sf.write(os.path.join(audio_dir, f"audio_{i}_value_{row[4]:.3f}.wav"), x, ftm.constants["sr"])