import pywt
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from joblib import Parallel, delayed

# db4 Decomposition Filter Coefficients
h_low = [
    -0.01059740, 0.03288301, 0.03084138, -0.18703481, 
    -0.02798376, 0.63088076, 0.71484657, 0.23037781
]

# The High-Pass filter is derived from the Low-Pass by reversing 
# the order and alternating the signs (Quadrature Mirror Filter)
h_high = [h_low[i] * (-1)**i for i in range(len(h_low))][::-1]

# Filtres de reconstruction (Synthesis filters)
g_low = h_low[::-1]
g_high = h_high[::-1]



def denoise_1d_signal(signal, level=3):
    """Décomposition et reconstruction manuelle pour un canal."""
    # 1. Décomposition
    coeffs = full_manual_dwt(signal, levels=level)
    
    # 2. Reconstruction (en ignorant le dernier niveau de détail D1)
    # Rappel : coeffs = [A3, D3, D2, D1]
    a3, d3, d2, d1 = coeffs
    
    a2_rec = manual_idwt_step(a3, d3)
    a1_rec = manual_idwt_step(a2_rec, d2)
    final_signal = manual_idwt_step(a1_rec, None) # None pour supprimer le bruit D1
    
    return final_signal

def parallel_denoise(X, n_jobs=-1):
    """Applique le débruitage sur tous les canaux en parallèle."""
    # X shape: (epochs, channels, times)
    n_epochs, n_channels, n_times = X.shape
    
    # On "aplatit" pour traiter chaque canal comme une unité indépendante
    X_flat = X.reshape(-1, n_times)
    
    # Exécution en parallèle
    results = Parallel(n_jobs=n_jobs)(
        delayed(denoise_1d_signal)(sig) for sig in X_flat
    )
    
    # On redonne la forme originale
    return np.array(results).reshape(n_epochs, n_channels, -1)

def manual_dwt_step(signal, h_low, h_high):
    """Applies one level of DWT to a 1D signal."""
    # 1. Filter the signal
    # 'valid' or 'same' padding keeps the sizes manageable
    approx = np.convolve(signal, h_low, mode='same')
    detail = np.convolve(signal, h_high, mode='same')
    
    # 2. Downsample (Keep every second point)
    # This is the 'D' in DWT (Discrete)
    approx = approx[::2]
    detail = detail[::2]
    
    return approx, detail

def full_manual_dwt(signal, levels=3):
    current_input = signal
    all_details = []
    
    for i in range(levels):
        approx, detail = manual_dwt_step(current_input, h_low, h_high)
        all_details.append(detail)
        current_input = approx # The new input for the next level
        
    # Result is: [Final_Approximation, Detail_L3, Detail_L2, Detail_L1]
    return [current_input] + all_details[::-1]


# Example of usage:
# a1, d1 = manual_dwt_step(eeg_channel_data, h_low, h_high)

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

def manual_idwt_step(approx, detail):
    """Reconstruit un niveau de signal à partir de l'approximation et du détail."""
    
    # 1. Upsampling : insérer des zéros
    def upsample(x):
        res = np.zeros(len(x) * 2)
        res[::2] = x
        return res

    a_up = upsample(approx)
    # Si on veut débruiter (votre but), on peut passer un 'detail' nul
    d_up = upsample(detail) if detail is not None else np.zeros(len(approx) * 2)

    # 2. Filtrage (Convolution)
    # On utilise 'same' pour garder la dimension cohérente
    low_part = np.convolve(a_up, g_low, mode='same')
    high_part = np.convolve(d_up, g_high, mode='same')
    
    # On détermine la longueur cible (celle du signal original à ce niveau)
    # Pour faire simple, on s'aligne sur la longueur de low_part
    n = min(len(low_part),len(high_part))
    high_part = high_part[:n]
    low_part = low_part[:n]
    # 3. Somme
    return low_part + high_part

def full_manual_idwt(coeffs):
    # coeffs est [A3, D3, D2, D1]
    current_approx = coeffs[0]
    details = coeffs[1:] # [D3, D2, D1]
    
    for i, detail in enumerate(details):
        # Si c'est le dernier niveau (D1), on applique le débruitage
        if i == len(details) - 1:
            current_approx = manual_idwt_step(current_approx, None)
        else:
            current_approx = manual_idwt_step(current_approx, detail)
            
    return current_approx

def extract_wavelet_reconstruction(data, wavelet, level):
    """
    Apply Wavelet transform and reconstruct the signal.
    This keeps the shape (epochs, channels, times) consistent for CSP.
    """

    wawelet = parallel_denoise(data, level)

    # Ensure the time dimension matches exactly (trimming if necessary)
    return wawelet[..., :data.shape[-1]]