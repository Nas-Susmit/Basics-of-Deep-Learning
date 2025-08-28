# forward_backward.py
import numpy as np
from utils import relu, sigmoid, relu_backward, sigmoid_backward, initialize_parameters_he

# Linear forward/back helpers
def linear_forward(A_prev, W, b):
    Z = W @ A_prev + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation, keep_prob=1.0, seed=None):
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "sigmoid":
        A, act_cache = sigmoid(Z)
    else:
        raise ValueError("Unknown activation")

    D = None
    if keep_prob < 1.0:
        if seed is not None:
            np.random.seed(seed)
        D = (np.random.rand(*A.shape) < keep_prob).astype(int)
        A = (A * D) / keep_prob

    cache = (lin_cache, act_cache, D, keep_prob)
    return A, cache

def L_model_forward(X, parameters, keep_probs=None, seed=None):
    """
    Forward pass for L-layer network. keep_probs: list for hidden layers.
    """
    caches = []
    A = X
    L = len(parameters) // 2
    if keep_probs is None:
        keep_probs = [1.0] * (L - 1)

    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        kp = keep_probs[l-1] if (l-1) < len(keep_probs) else 1.0
        A, cache = linear_activation_forward(A_prev, W, b, activation="relu", keep_prob=kp, seed=seed)
        caches.append(cache)

    # output layer
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A, W, b, activation="sigmoid", keep_prob=1.0)
    caches.append(cache)
    return AL, caches

# Backward helpers
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1.0 / m) * dZ @ A_prev.T
    db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, lambd=0.0):
    lin_cache, act_cache, D, keep_prob = cache
    # dropout mask
    if D is not None and keep_prob < 1.0:
        dA = (dA * D) / keep_prob

    if activation == "relu":
        dZ = relu_backward(dA, act_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, act_cache)
    else:
        raise ValueError("Unknown activation")

    dA_prev, dW, db = linear_backward(dZ, lin_cache)
    if lambd > 0.0:
        # add L2 regularization
        _, W, _ = lin_cache
        m = lin_cache[0].shape[1]
        dW += (lambd / m) * W
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, lambd=0.0):
    grads = {}
    L = len(caches)
    m = Y.shape[1]
    dAL = - (np.divide(Y, AL + 1e-12) - np.divide(1 - Y, 1 - AL + 1e-12))

    # output layer L
    current_cache = caches[-1]
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, activation="sigmoid", lambd=lambd)
    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # hidden layers l = L-1 ... 1
    for l in reversed(range(1, L)):
        current_cache = caches[l-1]
        dA = grads["dA" + str(l)]
        dA_prev, dW, db = linear_activation_backward(dA, current_cache, activation="relu", lambd=lambd)
        grads["dA" + str(l-1)] = dA_prev
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db

    return grads
