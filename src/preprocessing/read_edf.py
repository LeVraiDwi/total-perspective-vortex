import pywt
import numpy as np
import pyedflib
import mne
from mne.datasets import eegbci
from mne.datasets.eegbci import load_data
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.time_frequency import tfr_array_morlet

motor_channels = ['C3', 'Cz', 'C4']

def create_epochs(raw, events, run):
    task_id = {}

    if run in [3, 4, 7, 8, 11, 12]:
        task_id = {
            'left_fist': 1,  # T1 corresponds to left fist
            'right_fist': 2,  # T2 corresponds to right fist
        }
    elif run in [5, 6, 9, 10, 13, 14]:
        task_id = {
            'both_fists': 1,  # T1 corresponds to both fists
            'both_feet': 2,  # T2 corresponds to both feet
        }

    tmin, tmax = -1., 4.
    epochs = mne.Epochs(raw, events, task_id, tmin, tmax, proj=True,
                        baseline=None, preload=True)

    return epochs


def compute_time_frequency_analysis(epochs, subject: int, run: int):
    X_raw = epochs.get_data()
    y = epochs.events[:, -1]
    freqs = np.arange(2, 40, 2)
    n_cycles = np.maximum(1, freqs / 4.0)
    try:
        power = tfr_array_morlet(
            X_raw, sfreq=epochs.info["sfreq"],
            freqs=freqs, n_cycles=n_cycles,
            output="power",
            n_jobs=1
        )
    except ValueError as e:
        raise RuntimeError(f"Wavelet transform failed for subject {subject}, run {run}: {e}") from e
    return power.mean(axis=-1), y, freqs

def load_raw(subject: int, run: int):
    mne.set_log_level("WARNING")
    files = load_data(subject, run, path="/home/tcosse/total-perspective-vortex/data/physionet.org/files/eegmmidb/1.0.0/")
    if not files:
        raise RuntimeError(f"No data files found for subject {subject}, run {run}")
    raws = mne.io.read_raw_edf(files[0], preload=True, verbose="ERROR")
    eegbci.standardize(raws)
    raws.set_montage("standard_1005", on_missing="ignore")
    raws.plot(duration=5, n_channels=32)
    return raws
    
def extract_events(raw):
    events, event_id = mne.events_from_annotations(raw)
    if not event_id:
        raise RuntimeError("No events found in the raw data annotations")
    return events, event_id

def load_and_visualize_raw(subject: int, run: int):
    mne.set_log_level("WARNING")
    raw = load_raw(subject, run)

    return raw

def apply_filters(raw, l_freq=1., h_freq=40., visualize=True):
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    motor_ch_available = [ch for ch in motor_channels if ch in raw.ch_names]
    if not motor_ch_available:
        print(f"    No motor channels , skipping")

    raw = raw.copy().pick_channels(motor_ch_available)
    raw.filter(7, 32)
    if visualize:
        raw.plot(block=False, title="Filtered EEG Data (1-40 Hz)", show_scrollbars=False, scalings='auto')
        plt.show()
    raw.compute_psd(fmin=l_freq, fmax=h_freq, method="multitaper", verbose=False, n_jobs=1)

    return raw


def create_epochs_from_raw(raw, run: int, subject: int):
    events, event_id = extract_events(raw)
    epochs = create_epochs(raw, events, run)
    if len(epochs) == 0:
        raise RuntimeError(f"All epochs were dropped for subject {subject}, run {run}")
    return epochs

def compute_time_frequency_analysis(epochs, subject: int, run: int):
    X_raw = epochs.get_data()
    y = epochs.events[:, -1]
    freqs = np.arange(2, 40, 2)
    n_cycles = np.maximum(1, freqs / 4.0)
    try:
        power = tfr_array_morlet(
            X_raw, sfreq=epochs.info["sfreq"],
            freqs=freqs, n_cycles=n_cycles,
            output="power",
            n_jobs=1
        )
    except ValueError as e:
        raise RuntimeError(f"Wavelet transform failed for subject {subject}, run {run}: {e}") from e
    return power.mean(axis=-1), y, freqs


class WaveletTransformer:
    def __init__(self, frequencies, wavelet='morl', sfreq=160):
        self.frequencies = frequencies
        self.wavelet = wavelet
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_epochs, n_channels, _ = X.shape
        n_freqs = len(self.frequencies)

        scales = pywt.frequency2scale(
            self.wavelet,
            self.frequencies / self.sfreq
        )

        features = np.zeros((n_epochs, n_channels, n_freqs))

        for ep in range(n_epochs):
            for ch in range(n_channels):
                coeffs, _ = pywt.cwt(
                    X[ep, ch],
                    scales,
                    self.wavelet
                )
                power = np.abs(coeffs) ** 2
                features[ep, ch] = power.mean(axis=1)

        return features.reshape(n_epochs, -1)

def preprocess(subject: int, run: int, l_freq=1., h_freq=40., visualize=True):
    try:
        raw = load_and_visualize_raw(subject, run)
        
        raw = apply_filters(raw, l_freq, h_freq, visualize)

        epochs = create_epochs_from_raw(raw, run, subject)
        
        power_mean, y, freqs = compute_time_frequency_analysis(epochs, subject, run)
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed for subject {subject}, run {run}: {e}") from e


preprocess(1, 1)
#pipeline = Pipeline([
#    ("wavelet", WaveletTransformer(
#        frequencies=frequencies,
#        sfreq=sfreq
#    )),
#    ("scaler", StandardScaler()),
#    ("clf", SVC(
#        kernel="linear",
#        C=1.0
#    ))
#])
#
#cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#
#scores = cross_val_score(
#    pipeline,
#    X,
#    y,
#    cv=cv,
#    scoring="accuracy"
#)

print("Accuracy per fold:", scores)
print("Mean accuracy:", scores.mean())
