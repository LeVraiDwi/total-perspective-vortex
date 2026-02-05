from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.datasets.eegbci import load_data, standardize

HandTask = [3, 4, 7, 8, 11, 12]
BothTask = [5, 6, 9, 10, 13, 14]

# --- 1. PREPROCESSING & VISUALIZATION ---
def load_and_clean_data(subject: int, run: int, visualize: bool = False):
    # Load PhysioNet sample (Motor Imagery: Task 1 - Left vs Right Hand)
    # Note: Replace with actual local path to your PhysioNet .edf files
    files = load_data(subject, run, path="data/", base_url="https://physionet.org/files/eegmmidb/1.0.0/")
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)
    
    if 11 in HandTask:
        raw.annotations.rename(dict(T1="left_fist", T2="right_fist"))  # as documented on PhysioNet
    elif 11 in BothTask:
        raw.annotations.rename(dict(T1="both_fists", T2="both_feet"))  # as documented on PhysioNet
    else:
        raise "trouble"
    
    if visualize:
        print("Visualizing Raw Data...")
        raw.plot(n_channels=64, scalings='auto', title="Raw EEG")

    # Filter to keep Mu and Beta bands (8-30 Hz) - critical for Motor Imagery
    raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')
    
    if visualize:
        print("Visualizing Filtered Data...")
        raw.plot(n_channels=64, scalings='auto', title="Filtered EEG (8-30Hz)")
    
    return raw