# metrics.py
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

def predict(parameters, X, forward_func):
    AL, _ = forward_func(X, parameters, keep_probs=None)
    return (AL > 0.5).astype(int)

def evaluate(parameters, X, Y, forward_func, title=""):
    y_pred = predict(parameters, X, forward_func)
    acc = accuracy_score(Y.flatten(), y_pred.flatten())
    cm = confusion_matrix(Y.flatten(), y_pred.flatten())
    print(f"\n=== {title} ===")
    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print(classification_report(Y.flatten(), y_pred.flatten(), digits=4))
    return acc, cm
