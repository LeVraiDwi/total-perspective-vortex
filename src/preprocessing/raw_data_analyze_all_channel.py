import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Base path
base_path = Path("/home/tcosse/total-perspective-vortex/data/physionet.org/files/eegmmidb/1.0.0")

# Liste pour stocker PSD de tout le batch
all_psd = []
all_freqs = None

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
        clean_channel_names(raw)
        raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})
        raw.set_montage("standard_1005", on_missing="ignore")

        # Calcul PSD sur tous les canaux EEG, sans filtrage
        psd_obj = raw.compute_psd(fmin=0.5, fmax=60, n_fft=2048, n_overlap=1024)

        psd = psd_obj.get_data()  # shape = (n_channels, n_freqs)
        freqs = psd_obj.freqs

        # Moyenne sur tous les canaux du fichier
        psd_mean = psd.mean(axis=0)

        all_psd.append(psd_mean)
        if all_freqs is None:
            all_freqs = freqs

# Convertir en array numpy
all_psd = np.array(all_psd)

# Moyenne PSD sur tout le batch
psd_batch_mean = all_psd.mean(axis=0)

# Affichage
plt.figure(figsize=(8,4))
plt.plot(all_freqs, 10*np.log10(psd_batch_mean))
#plt.plot(all_freqs, psd_batch_mean)
plt.title("PSD moyenne batch EEGMMIDB (tous les canaux)")
plt.xlabel("Fréquence (Hz)")
#plt.ylabel("Power (V²/Hz)")
plt.ylabel("Power (dB)")
plt.xlim(0.5, 60)
plt.grid(True)
plt.show()
