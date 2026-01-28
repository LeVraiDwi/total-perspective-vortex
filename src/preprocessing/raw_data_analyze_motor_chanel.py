import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Base path EEGMMIDB
base_path = Path("/home/tcosse/total-perspective-vortex/data/physionet.org/files/eegmmidb/1.0.0")


# Liste pour stocker PSD des canaux moteurs filtrés
all_psd_motor = []
all_freqs_motor = None

def clean_channel_names(raw):
    mapping = {ch: ch.replace('.', '').upper() for ch in raw.ch_names}
    raw.rename_channels(mapping)

for subj_folder in sorted(base_path.glob("S*")):
    subject_id = subj_folder.name
    print(f"Processing {subject_id}")

    for edf_file in sorted(subj_folder.glob("*.edf")):
        run_id = edf_file.stem
        print(f"  Loading {run_id}")

        raw = mne.io.read_raw_edf(edf_file, preload=True)

        # Nettoyage noms de canaux
        clean_channel_names(raw)
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        raw.set_montage("standard_1005", on_missing="ignore")

        # Filtrage des canaux moteurs : mu/beta band (8–30 Hz)
        raw.filter(l_freq=0, h_freq=40)

        # Calcul PSD sur les canaux moteurs filtrés
        psd_obj = raw.compute_psd( n_fft=2048, n_overlap=1024)
        psd = psd_obj.get_data()  # shape = (n_channels, n_freqs)
        freqs = psd_obj.freqs

        # Moyenne sur les canaux moteurs
        psd_mean_motor = psd.mean(axis=0)
        all_psd_motor.append(psd_mean_motor)

        if all_freqs_motor is None:
            all_freqs_motor = freqs

# Convertir en array numpy
all_psd_motor = np.array(all_psd_motor)

# PSD moyenne batch canaux moteurs
psd_motor_batch_mean = all_psd_motor.mean(axis=0)

# Affichage
plt.figure(figsize=(8,4))
plt.plot(all_freqs_motor, 10*np.log10(psd_motor_batch_mean))
plt.title("PSD moyenne batch EEGMMIDB (canaux moteurs C3, CZ, C4, filtrés 0–40 Hz)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Power (dB)")
plt.grid(True)
plt.show()
