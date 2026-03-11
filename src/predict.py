import mne
import joblib
import numpy as np
from preprocessing.load_data import load_and_clean_data

def Predict(subject: int, run: int, model_path: str):
    # 1. Charger le modèle sauvegardé
    print(f"Chargement du modèle : {model_path}")
    clf = joblib.load(model_path)
    
    # 2. Charger et nettoyer les nouvelles données (Inférence)
    # On utilise la même fonction de nettoyage pour garder la cohérence (8-30Hz)
    raw = load_and_clean_data(subject, run, visualize=False)
    
    # 3. Créer les Epochs
    events, event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=4.0, 
                        baseline=None, preload=True)
    
    # Données X: (n_epochs, n_channels, n_times)
    X_new = epochs.get_data()
    y_true = epochs.events[:, -1] # Pour vérifier si la prédiction est juste

    # 4. Exécuter la prédiction
    # Le pipeline s'occupe de WaveletDenoiseTransformer + CSP + LDA automatiquement
    predictions = clf.predict(X_new)
    
    # 5. Affichage des résultats
    print("-" * 30)
    print(f"Résultats pour Sujet {subject}, Run {run}:")
    
    for i, p in enumerate(predictions):
        label_pred = [k for k, v in event_id.items() if v == p][0]
        label_true = [k for k, v in event_id.items() if v == y_true[i]][0]
        match = "✅" if p == y_true[i] else "❌"
        print(f"Epoch {i}: Prédit = {label_pred} | Réel = {label_true} {match}")
    
    accuracy = np.mean(predictions == y_true)
    print(f"\nPrécision globale sur ce run : {accuracy:.4f}")

if __name__ == "__main__":
    # Exemple : Prédire sur le sujet 4, run 12 avec le modèle entraîné précédemment
    Predict(subject=7, run=3, model_path="./model/Export_7_11.pkl")