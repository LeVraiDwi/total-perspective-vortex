import pywt
import matplotlib.pyplot as plt
import numpy as np
from mne import EvokedArray

def plot_topoMap(clf, X_raw, y, epochs):
    # 1. Extract patterns: shape is (n_components, n_channels) -> (4, 64)
    # We want to treat these 4 components as 4 "time points" for plotting
    X_wavelet = extract_wavelet_reconstruction_plot(X_raw)

    clf.fit(X_wavelet, y)
    patterns_data = clf.named_steps['CSP'].patterns_.T

    # 2. IMPORTANT: The data must be (n_channels, n_points)
    # Since patterns_data is (64, 4), this matches!
    patterns_to_plot = EvokedArray(patterns_data, epochs.info)
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

def plot_scalogram(data, fs=160):
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