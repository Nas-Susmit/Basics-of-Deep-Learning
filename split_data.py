# split_data.py
import numpy as np
from sklearn.model_selection import train_test_split

def train_dev_test_split(X, Y, test_size=0.15, dev_size=0.15, random_state=3, stratify=True):
    """
    Splits X (n_x, m) and Y (1, m) into train/dev/test (transposed for sklearn compatibility).
    Returns: X_train, Y_train, X_dev, Y_dev, X_test, Y_test (all shapes maintained as (n_x, m_part) and (1, m_part))
    """
    X_t = X.T
    Y_t = Y.flatten()
    total_test_dev = test_size + dev_size
    strat = Y_t if stratify else None

    X_train, X_temp, y_train, y_temp = train_test_split(X_t, Y_t,
                                                        test_size=total_test_dev,
                                                        random_state=random_state,
                                                        stratify=strat)
    rel_dev = dev_size / total_test_dev if total_test_dev > 0 else 0
    if rel_dev > 0:
        X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp,
                                                        test_size=1 - rel_dev,
                                                        random_state=random_state,
                                                        stratify=y_temp if stratify else None)
    else:
        X_dev, y_dev = np.empty((0, X_t.shape[1])), np.empty((0,))
        X_test, y_test = X_temp, y_temp

    # Transpose back
    return (X_train.T, y_train.reshape(1, -1),
            X_dev.T,   y_dev.reshape(1, -1),
            X_test.T,  y_test.reshape(1, -1))
