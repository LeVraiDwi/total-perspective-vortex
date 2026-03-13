import mne
import joblib
import numpy as np
from load_data import load_and_clean_data
from sklearn.model_selection import train_test_split
import time

def Predict(subject: int, run: int, model_path = "./model/", max_delay = 2.0, verbose = True, delay = 0.1):
    # 1. Charger le modèle sauvegardé
    path = f"{model_path}Export_{subject}_{run}.pkl"
    #print(f"Chargement du modèle : {path}")
    try:
        data = joblib.load(path)
        clf = data["clf"]
        random_state = data["random_state"]
        test_size = data["test_size"]
    except FileNotFoundError:
        print(f"Model file not found: {path}")
        return
    
    # 2. Charger et nettoyer les nouvelles données (Inférence)
    # On utilise la même fonction de nettoyage pour garder la cohérence (8-30Hz)
    raw = load_and_clean_data(subject, run, visualize=False)
    
    # 3. Créer les Epochs
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=4.0, 
                        baseline=None, preload=True)
    
    # Données X: (n_epochs, n_channels, n_times)
    X_raw = epochs.get_data()
    y_true = epochs.events[:, -1] # Pour vérifier si la prédiction est juste
    X_train_val, X_validation, y_train_val, y_validation = train_test_split(X_raw, y_true, test_size=test_size, random_state=random_state)

    correct = 0
    processing_times = []

    for  i, (x_epoch, true_label) in enumerate(zip(X_validation, y_validation)):
        start_time = time.time()
        pred = clf.predict(x_epoch.reshape(1, x_epoch.shape[0], x_epoch.shape[1]))
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        if processing_time > max_delay:
            print(f"⚠️  WARNING: Processing time {processing_time:.3f}s exceeds {max_delay}s limit!")
        
        is_equal = pred == true_label
        if is_equal:
            correct += 1
        
        if verbose:
            status = "✅" if is_equal else "❌"
            print(f"epoch {i:02d}: [{pred}] [{true_label}] {status} ({processing_time:.3f}s)")
        
        time.sleep(delay)

        accuracy = correct / len(y_validation)
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)

    if verbose:
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{len(y_validation)})")
        print(f"Average processing time: {avg_processing_time:.3f}s")
        print(f"Maximum processing time: {max_processing_time:.3f}s")
        print(f"Samples exceeding {max_delay}s limit: {sum(1 for t in processing_times if t > max_delay)}")
    return accuracy