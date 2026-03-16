import mne
import numpy as np
from csp import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit , cross_val_score
import joblib
from visualize import plot_scalogram
from WaveletDenoiseTransformer import WaveletDenoiseTransformer
from load_data import load_and_clean_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

tmin, tmax = 0.0, 4.0
subjects = 4
runs = 14# motor imagery: hands vs feet

def save_model(model_data, subject, run):
    model_path = f"./model/Export_{subject}_{run}.pkl"
    joblib.dump(model_data, model_path)
    #print(f"Model saved as Export_{subject}_{run}.pkl")
    return

def cross_val_sc(clf, X, y, random_state):
    cv = ShuffleSplit(10, test_size=0.2, random_state=random_state)

    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None, verbose=0)
    scores_str = np.array2string(scores, precision=4, separator=' ')
    
    # Combine into your specific format
    print(f"{scores_str}\ncross_val_score: {np.mean(scores):.4f}")
    return scores

def score_model(clf, X, y):
    # Score sur le Validation Set (pour ajuster tes idées)
    val_score = clf.score(X, y)
    print(f"Validation Accuracy: {val_score:.4f}")
    
    return val_score

# --- 3. Train TREATMENT PIPELINE ---
def Train(subject: int, run: int, random_state = 42, visualize: bool = False, test_size = 0.3, cross_val = True):
    mne.set_log_level('WARNING')
    try:
        raw = load_and_clean_data(subject, run, visualize)
    except Exception:
        return
    # Create Epochs (tmin/tmax adjusted for PhysioNet MI tasks)
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, 
                        baseline=None, preload=True)
    
    # Get labels and data
    X_raw = epochs.get_data() # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]

    # Apply Wavelet Transform to reduce noise/temporal redundancy
    if visualize:
        plot_scalogram(X_raw)
    
    X_train_val, X_validation, y_train_val, y_validation = train_test_split(X_raw, y, test_size=test_size, random_state=random_state, stratify=y)
    # Setup the Processing Pipeline: CSP + LDA
    # CSP reduces spatial dimensions, LDA makes the final decision
    clf = Pipeline([
        ('wavelet', WaveletDenoiseTransformer(wavelet='db4', level=3)),
        ('CSP', CSP(n_components=4)),
        ('LDA', LinearDiscriminantAnalysis()),
    ], verbose=False)

    clf.fit(X_train_val, y_train_val)
   
    if cross_val:
        scores = cross_val_sc(clf, X_train_val, y_train_val, random_state=random_state)

    model_data = {
        "clf": clf,
        "random_state": random_state,
        "test_size": test_size
    }
    
    save_model(model_data, subject, run)
    return