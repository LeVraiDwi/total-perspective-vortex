import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from mne.datasets.eegbci import load_data
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP, get_spatial_filter_from_estimator
from mne.io import concatenate_raws, read_raw_edf
import pywt

print(__doc__)
motor_channels = ['C3', 'Cz', 'C4']
tmin, tmax = 0.0, 4.0
subjects = 4
runs = 11# motor imagery: hands vs feet
HandTask = [3, 4, 7, 8, 11, 12]
BothTask = [5, 6, 9, 10, 13, 14]
# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
def load_Raw(subject: int, run: int):
    files = load_data(subject, run, path="/home/tcosse/total-perspective-vortex/data/physionet.org/files/eegmmidb/1.0.0/")
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)
    
    if run in HandTask:
        event_id = ["left_fist", "right_fist"]
        event_name = dict(T1="left_fist", T2="right_fist")
    elif run in BothTask:
        event_id = ["both_fists", "both_feet"]
        event_name = dict(T1="both_fists", T2="both_feet")
    else:
        raise "trouble"
    raw.annotations.rename(event_name)  # as documented on PhysioNet
    motor_ch_available = [ch for ch in motor_channels if ch in raw.ch_names]
    if not motor_ch_available:
        print(f"    No motor channels , skipping")
    
    raw.filter(0.0, 40.0, fir_design="firwin", skip_by_annotation="edge")
    #raw = raw.copy().pick_channels(motor_ch_available)
    return raw, event_id

# 3. Wavelet Feature Extraction
def extract_wavelet_features(data):
    # Using 'db4' wavelet as an example
    coeffs = pywt.wavedec(data, 'db4', level=3, axis=-1)
    # Flattening coefficients to create a feature vector per epoch
    return np.concatenate([c.reshape(c.shape[0], -1) for c in coeffs], axis=-1)

def filter_data(raw):
    # Apply band-pass filter
    ret = extract_wavelet_features(raw)
    return ret

def read_epochs(raw, event_id):
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = Epochs(
        raw,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )

    epochs_train = epochs.copy().crop(tmin, tmax)
    labels = epochs.events[:, -1] - 2
    return epochs, epochs_train, labels
    
def process(raw, epochs, epochs_train, labels):
    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data(copy=False)
    epochs_data_train = epochs_train.get_data(copy=False)
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1.0 - class_balance)
    print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

    # plot eigenvalues and patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)
    spf = get_spatial_filter_from_estimator(csp, info=epochs.info)
    spf.plot_scree()
    spf.plot_patterns(components=np.arange(2))

    sfreq = raw.info["sfreq"]
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
    
    scores_windows = []
    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)
        # Plot scores over time
    w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
    plt.axvline(0, linestyle="--", color="k", label="Onset")
    plt.axhline(0.5, linestyle="-", color="k", label="Chance")
    plt.xlabel("time (s)")
    plt.ylabel("classification accuracy")
    plt.title("Classification score over time")
    plt.legend(loc="lower right")
    plt.show()

def train(subject: int, run: int):
    raw, event_id = load_Raw(subject, run)
    # Reshape for Wavelet (Combine channel and time/freq info)
    # We apply wavelet to each channel and flatten
    X_wavelet = np.array([extract_wavelet_features(epoch) for epoch in X])
    X_features = X_wavelet.reshape(X_wavelet.shape[0], -1) # Flatten to (n_samples, n_features)
    epochs, epochs_train, labels = read_epochs(raw, event_id)
    process(raw, epochs, epochs_train, labels)
    return

train(subjects, runs)