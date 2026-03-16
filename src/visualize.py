import pywt
import matplotlib.pyplot as plt
import numpy as np
from mne import EvokedArray

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