# visualize.py
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model_fn, X, Y, title="Decision Boundary"):
    """
    model_fn: function that accepts X_grid (2, N) and returns predictions shape (1, N)
    X: (2, m)
    Y: (1, m)
    """
    X_np = X.T
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()].T  # shape (2, N)
    Z = model_fn(grid)                       # expects (1, N)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=Y.flatten(), edgecolors='k', cmap=plt.cm.Spectral)
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.show()

def plot_learning_curve(costs, interval=100, title="Learning curve"):
    epochs = np.arange(0, len(costs)) * interval
    plt.figure()
    plt.plot(epochs, costs)
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.title(title)
    plt.show()
