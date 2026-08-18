from pnp_synth.physical import ftm
import soundfile as sf
import os
import numpy as np

min_p, max_p = 0.15, 2

omega_A, tau_A, p_A, D_A, alpha_A = 3000, 0.5, min_p, 0.05, 1.0
log_omega_A, logp_A, logD_A = np.log10(omega_A), np.log10(p_A), np.log10(D_A)

omega_B, tau_B, p_B, D_B, alpha_B = 3000, 0.05, max_p, 0.05, 1.0
log_omega_B, logp_B, logD_B = np.log10(omega_B), np.log10(p_B), np.log10(D_B)

audio_dir = "generations/test_log/audios" # os.path.join(save_dir, "x")
os.makedirs(audio_dir, exist_ok=True)

A = [omega_A, tau_A, p_A, D_A, alpha_A]
log_A = [log_omega_A, tau_A, logp_A, logD_A, alpha_A]
B = [omega_B, tau_B, p_B, D_B, alpha_B]
log_B = [log_omega_B, tau_B, logp_B, logD_B, alpha_B]

def synth(audio_name: str, theta: list[float], logscale: bool):
    print(f"Processing row {audio_name}")
    # Physical audio synthesis (g). theta -> x
    x = ftm.rectangular_drum(theta, logscale, **ftm.constants).cpu()
    x = x / max(x)

    sf.write(os.path.join(audio_dir, f"audio_{audio_name}.wav"), x, ftm.constants["sr"])

synth(audio_name = "A", theta = A, logscale = False)
synth(audio_name = "B", theta = B, logscale = False)
synth(audio_name = "log_A", theta = log_A, logscale = True)
synth(audio_name = "log_B", theta = log_B, logscale = True)

AB = np.linspace(A, B, 7)[1:-1]
for i, theta in enumerate(AB):
    synth(f"AB_{i+1}", theta, False)

log_AB = np.linspace(log_A, log_B, 7)[1:-1]
for i, theta in enumerate(log_AB):
    synth(f"log_AB_{i+1}", theta, True)