from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, y):
        fitValue = CSP_fit(X, y)
        self.fit_value = np.vstack(list(fitValue.values()))
        return self
    
    def transform(self, X):
        # On appelle votre fonction existante ici
        return CSP_transform(X, self.fit_value)



def CSP_transform(X, fit_value):
    # 1. Project all epochs at once: (Epochs, Filters, Time)
    # This tells NumPy to apply the filter matrix to the last two dims of every epoch
    Z = np.matmul(fit_value, X)

    # 2. Calculate variance along the time axis (axis=-1)
    variances = np.var(Z, axis=-1)

    # 3. Apply log
    features = np.log(variances)
    return features

def CSP_fit(X, y):
    FitValue = {}
    for label in np.unique(y):
        cl = X[y == label]
        not_cl = X[y != label]
        
        # 2. Function/Logic to get mean covariance for target
        cov_target = compute_mean_covariance(cl)
    
        # 3. Function/Logic to get mean covariance for everything else
        cov_others = compute_mean_covariance(not_cl)

        u, s, vh = np.linalg.svd(cov_target + cov_others)
        p = np.diag(1.0 / (np.sqrt(s))) @ u.T
        S_target = p @ cov_target @ p.T
        u_rot, s_rot, vh_rot = np.linalg.svd(S_target)
        
        W = u_rot.T @ p

        FitValue[label] = np.vstack((W[:2], W[-2:]))
    return FitValue

def compute_mean_covariance(cl):
    CovarSum = 0.0
    for trial in cl:
        trial_cov = np.dot(trial, trial.T)
        normalize_cov = trial_cov / np.trace(trial_cov)\
        
        CovarSum += normalize_cov
    return CovarSum / len(cl)