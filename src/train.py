import mne
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
import joblib
from preprocessing.visualize import plot_scalogram, plot_topoMap
from preprocessing.waweletTransformer import WaveletDenoiseTransformer
from preprocessing.load_data import load_and_clean_data
import argparse

tmin, tmax = 0.0, 4.0
subjects = 4
runs = 14# motor imagery: hands vs feet

# --- 3. Train TREATMENT PIPELINE ---
def Train(subject: int, run: int, visualize: bool = False):
    mne.set_log_level('WARNING')
    raw = load_and_clean_data(subject, run, visualize)
    
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
    
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    # Setup the Processing Pipeline: CSP + LDA
    # CSP reduces spatial dimensions, LDA makes the final decision
    clf = Pipeline([
        ('wavelet', WaveletDenoiseTransformer(wavelet='db4', level=3)),
        ('CSP', CSP(n_components=4, reg=None, log=False, norm_trace=False)),
        ('LDA', LinearDiscriminantAnalysis()),
    ])
    scores = cross_val_score(clf, X_raw, y, cv=cv, n_jobs=None, verbose=0)
    
    joblib.dump(clf, f"./model/Export_{subject}_{run}.pkl")
    print(f"Model saved as Export_{subject}_{run}.pkl")
    
    # --- CUSTOM CONSOLE OUTPUT ---
    # Format scores array to a clean string
    # precision=4 limits the decimals, separator=' ' removes commas
    scores_str = np.array2string(scores, precision=4, separator=' ')
    
    # Combine into your specific format
    print(f"{scores_str}\ncross_val_score: {np.mean(scores):.4f}")
    #class_balance = np.mean(y == y[0])
    #class_balance = max(class_balance, 1.0 - class_balance)
    #print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

    if visualize:
        plot_topoMap(clf, X_raw, y, epochs)

def main():
    parser = argparse.ArgumentParser(
                    prog='Train',
                    description='Train a model on EEG data',
                    epilog='')
    parser.add_argument('subject', type=int, help="Subject number (int)")
    parser.add_argument('run', type=int, help="Run number (int)")
    parser.add_argument('-v', '--visualize',
                    action='store_true', help="print graph ?")
    args = parser.parse_args()
    Train(args.subject, args.run, args.visualize)

if __name__ == "__main__":
    main()
    print("end")