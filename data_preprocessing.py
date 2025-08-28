# data_preprocessing.py
import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def make_dataset(n_samples=1000, noise=0.2, random_state=3):
    """
    Generates and scales a 2D toy dataset (moons).
    Returns X (n_x, m), Y (1, m), scaler object.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    # Basic cleaning
    mask = np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    X = X.T  # shape (n_x, m)
    Y = y.reshape(1, -1)  # shape (1, m)
    return X, Y, scaler
