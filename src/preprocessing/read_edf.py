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
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne.time_frequency import tfr_array_morlet
from mne.io import concatenate_raws, read_raw_edf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

motor_channels = ['C3', 'Cz', 'C4']
tmin, tmax = -1.0, 4.0
subjects = 1
runs = [3, 4, 7, 8, 11, 12]  # motor imagery: hands vs feet

def create_epochs(raw, events, run):
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    
    tmin, tmax = -1., 4.
    epochs = mne.Epochs(
    raw,
    events,
    tmin=tmin,
    tmax=tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
    )
    epochs_train = epochs.copy()#.crop(tmin=1.0, tmax=2.0)
    labels = epochs.events[:, -1] - 2
    
    raw.plot(duration=5, n_channels=3)

    return epochs, epochs_train, labels


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

def load_raw(subject: int, run: list[int]):
    mne.set_log_level("WARNING")
    raw_fnames = load_data(subject, run, path="/home/tcosse/total-perspective-vortex/data/physionet.org/files/eegmmidb/1.0.0/")
    if not raw_fnames:
        raise RuntimeError(f"No data files found for subject {subject}, run {run}")
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)
    
    raw.set_montage("standard_1005", on_missing="ignore")
    if run == [3, 4, 7, 8, 11, 12]:
        raw.annotations.rename(dict(T1="left_fist", T2="right_fist"))  # as documented on PhysioNet
    elif run == [5, 6, 9, 10, 13, 14]:
        raw.annotations.rename(dict(T1="both_fists", T2="both_feet"))  # as documented on PhysioNet
    raw.set_eeg_reference(projection=True)
    raw.plot(duration=5, n_channels=32)
    return raw
    
def extract_events(raw):
    events, event_id = mne.events_from_annotations(raw)
    if not event_id:
        raise RuntimeError("No events found in the raw data annotations")
    return events, event_id

def load_and_visualize_raw(subject: int, run: list[int]):
    mne.set_log_level("WARNING")
    raw = load_raw(subject, run)

    return raw

def apply_filters(raw, l_freq=1., h_freq=40., visualize=True):
    motor_ch_available = [ch for ch in motor_channels if ch in raw.ch_names]

    if not motor_ch_available:
        print(f"    No motor channels , skipping")

    raw = raw.copy().pick_channels(motor_ch_available)
    raw.filter(l_freq, h_freq, fir_design="firwin", skip_by_annotation="edge")
    if visualize:
        raw.plot(block=False, title="Filtered EEG Data (1-40 Hz)", show_scrollbars=False, scalings='auto')
        plt.show()

    return raw


def create_epochs_from_raw(raw, run: int, subject: int):
    events, event_id = extract_events(raw)
    epochs, epochs_train, labels = create_epochs(raw, events, run)
    if len(epochs) == 0:
        raise RuntimeError(f"All epochs were dropped for subject {subject}, run {run}")
    return epochs, epochs_train, labels

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


def preprocess(subject: int, run: list[int], l_freq=1., h_freq=40., visualize=True):
    try:
        raw = load_and_visualize_raw(subject, run)
        
        raw = apply_filters(raw, l_freq, h_freq, visualize)

        epochs, epochs_train, labels = create_epochs_from_raw(raw, run, subject)
        
       # 2. Epoching
        epochs, epochs_train, labels = create_epochs_from_raw(raw, run, subject)
        epochs_data = epochs.get_data(copy=True)
        sfreq = epochs.info['sfreq']
        
        # 3. Define Cross-Validation
        cv = ShuffleSplit(10, test_size=0.2, random_state=42)

        # Assemble a classifier
        #csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)
        #lda = LinearDiscriminantAnalysis()
        #csp_pipe = Pipeline([("CSP", csp), ("LDA", lda)])
        
        #csp_scores = cross_val_score(csp_pipe, epochs_data, labels, cv=cv)

        # Assuming WaveletTransformer is defined as in your snippet
        wavelet_freqs = np.arange(8, 31, 2) 
        wavelet_pipe = Pipeline([
            ("wavelet", WaveletTransformer(frequencies=wavelet_freqs, sfreq=sfreq)),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear", C=1.0))
        ])

        # Running cross-validation for Wavelets
        wavelet_scores = cross_val_score(wavelet_pipe, epochs_data, labels, cv=cv)
        print(f"Wavelet Mean Accuracy: {wavelet_scores.mean():.2f}")

        ## 4. Reporting
        print(f"\n--- Results for Subject {subject} ---")
        #print(f"CSP + LDA Accuracy:     {np.mean(csp_scores):.3f}")
        print(f"Wavelet + SVC Accuracy: {np.mean(wavelet_scores):.3f}")
        # plot eigenvalues and patterns estimated on full data for visualization
        csp.fit(epochs_data, labels)
        csp.plot_patterns(epochs.info, title="CSP Patterns")

        # Compute for return values
        power_mean, y, freqs = compute_time_frequency_analysis(epochs, subject, run[0])
        return power_mean, y, freqs, csp_scores, wavelet_scores
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed for subject {subject}, run {run}: {e}") from e


preprocess(1, runs)
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
