import mne
import numpy as np
import pywt
import matplotlib.pyplot as plt
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from mne.datasets.eegbci import load_data
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin

motor_channels = ['C3', 'Cz', 'C4']
tmin, tmax = 0.0, 4.0
subjects = 4
runs = 14# motor imagery: hands vs feet
HandTask = [3, 4, 7, 8, 11, 12]
BothTask = [5, 6, 9, 10, 13, 14]

class WaveletDenoiseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        # Pour les ondelettes, il n'y a rien à "apprendre" sur le dataset,
        # donc on se contente de retourner self.
        return self
    
    
    def transform(self, X):
        # On appelle votre fonction existante ici
        return extract_wavelet_reconstruction(X, self.wavelet, self.level)
    
    


# --- 1. PREPROCESSING & VISUALIZATION ---
def load_and_clean_data(subject: int, run: int):
    # Load PhysioNet sample (Motor Imagery: Task 1 - Left vs Right Hand)
    # Note: Replace with actual local path to your PhysioNet .edf files
    files = load_data(subject, run, path="/home/tcosse/total-perspective-vortex/data/physionet.org/files/eegmmidb/1.0.0/")
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)
    
    if 11 in HandTask:
        raw.annotations.rename(dict(T1="left_fist", T2="right_fist"))  # as documented on PhysioNet
    elif 11 in BothTask:
        raw.annotations.rename(dict(T1="both_fists", T2="both_feet"))  # as documented on PhysioNet
    else:
        raise "trouble"
    
    print("Visualizing Raw Data...")
    raw.plot(n_channels=64, scalings='auto', title="Raw EEG")

    # Filter to keep Mu and Beta bands (8-30 Hz) - critical for Motor Imagery
    raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')
    
    print("Visualizing Filtered Data...")
    raw.plot(n_channels=64, scalings='auto', title="Filtered EEG (8-30Hz)")
    
    return raw

# --- 2. FEATURE EXTRACTION (WAVELETS) ---
def apply_wavelet_transform(epochs_data):
    """
    Apply Discrete Wavelet Transform (DWT).
    We use 'db4' (Daubechies) as it resembles EEG spikes/patterns.
    """
    # coeffs[0] = Approximation (Low freq), coeffs[1] = Detail (High freq)
    coeffs = pywt.wavedec(epochs_data, 'db4', level=3, axis=-1)
    
    # We use the Approximation coefficients as our 'cleaned' signal features
    return coeffs[0] 

def extract_wavelet_reconstruction(data, wavelet, level):
    """
    Apply Wavelet transform and reconstruct the signal.
    This keeps the shape (epochs, channels, times) consistent for CSP.
    """
    # Using Daubechies 4 wavelet
    wavelet = 'db4'
    level = 3

    # Apply DWT
    coeffs = pywt.wavedec(data, wavelet, level=level, axis=-1)

    # To keep CSP happy, we use the Approximation coefficients (low-freq)
    # and reconstruct them back to the original time dimension
    reconstructed = pywt.waverec(coeffs[:-1] + [None], wavelet, axis=-1)

    # Ensure the time dimension matches exactly (trimming if necessary)
    return reconstructed[..., :data.shape[-1]]

def extract_wavelet_reconstruction_plot(data):
    """
    Apply Wavelet transform and reconstruct the signal.
    This keeps the shape (epochs, channels, times) consistent for CSP.
    """
    # Using Daubechies 4 wavelet
    wavelet = 'db4'
    level = 3
    
    # Apply DWT
    coeffs = pywt.wavedec(data, wavelet, level=level, axis=-1)
    
    # To keep CSP happy, we use the Approximation coefficients (low-freq)
    # and reconstruct them back to the original time dimension
    reconstructed = pywt.waverec(coeffs[:-1] + [None], wavelet, axis=-1)
    
    plt.figure(figsize=(10, 6))
    names = ['Approximation'] + [f'Detail {i}' for i in range(level, 0, -1)]
    
    # Calculate energy (density) for each level
    energies = [np.sum(np.square(c)) for c in coeffs]
    
    plt.bar(names, energies, color='skyblue')
    plt.title('Wavelet Energy Density per Level')
    plt.ylabel('Energy (Power)')
    plt.show()

    # Ensure the time dimension matches exactly (trimming if necessary)
    return reconstructed[..., :data.shape[-1]]

def plot_scalogram(data, fs=160):
    import signal
    sample_data = data[0, 0, :] # First epoch, first channel
    
    # Continuous Wavelet Transform for a smooth "density" plot
    widths = np.arange(1, 31)
    cwtmatr, freqs = pywt.cwt(sample_data, widths, 'mexh')
    
    plt.figure(figsize=(10, 4))
    plt.imshow(np.abs(cwtmatr), extent=[0, 4, 1, 31], cmap='jet', aspect='auto',
               vmax=np.abs(cwtmatr).max(), vmin=-np.abs(cwtmatr).max())
    plt.title('Wavelet Specter (Scalogram) - Density over Time')
    plt.ylabel('Scale (Frequency)')
    plt.xlabel('Time (sec)')
    plt.colorbar(label='Intensity')
    plt.show()

# --- 3. Train TREATMENT PIPELINE ---
def Train(subject: int, run: int):
    raw = load_and_clean_data(subject, run)
    
    # Create Epochs (tmin/tmax adjusted for PhysioNet MI tasks)
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, 
                        baseline=None, preload=True)
    
    # Get labels and data
    X_raw = epochs.get_data() # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]

    # Apply Wavelet Transform to reduce noise/temporal redundancy
    X_wavelet = extract_wavelet_reconstruction_plot(X_raw)
    plot_scalogram(X_raw)
    
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    # Setup the Processing Pipeline: CSP + LDA
    # CSP reduces spatial dimensions, LDA makes the final decision
    clf = Pipeline([
        ('wavelet', WaveletDenoiseTransformer(wavelet='db4', level=3)),
        ('CSP', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ('LDA', LinearDiscriminantAnalysis()),
    ], verbose=False)
    scores = cross_val_score(clf, X_raw, y, cv=cv, n_jobs=None, verbose=False)
    class_balance = np.mean(y == y[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

    # 1. Extract patterns: shape is (n_components, n_channels) -> (4, 64)
    # We want to treat these 4 components as 4 "time points" for plotting
    clf.fit(X_wavelet, y)
    patterns_data = clf.named_steps['CSP'].patterns_.T

    # 2. IMPORTANT: The data must be (n_channels, n_points)
    # Since patterns_data is (64, 4), this matches!
    patterns_to_plot = mne.EvokedArray(patterns_data, epochs.info)
    actual_times = patterns_to_plot.times[:4]
    # 4. Plot using these calculated times
    print(f"Plotting CSP Patterns at times: {actual_times}")
    # 3. Plot
    # We use 'times' to pick which component to show (0, 1, 2, or 3)
    patterns_to_plot.plot_topomap(times=actual_times, 
                                  ch_type='eeg', 
                                  ncols=4, 
                                  nrows=1, 
                                  colorbar=True, 
                                  units='Patterns (AU)')
    plt.show()


if __name__ == "__main__":
    Train(subjects, runs)
    print("end")