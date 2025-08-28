# utils.py
import numpy as np

# --- Activations ---
def sigmoid(Z):
    A = 1.0 / (1.0 + np.exp(-Z))
    return A, Z

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def sigmoid_backward(dA, cacheZ):
    Z = cacheZ
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def relu_backward(dA, cacheZ):
    Z = cacheZ
    dZ = dA * (Z > 0)
    return dZ

# --- Initialization (He) ---
def initialize_parameters_he(layer_dims, seed=3):
    """
    layer_dims: list [n_x, n_h1, ..., n_y]
    returns parameters dict with W1..WL and b1..bL
    """
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims) - 1
    for l in range(1, L + 1):
        parameters["W" + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l-1]) *
                                     np.sqrt(2.0 / layer_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

# --- Dictionary/vector helpers for grad-check ---
def dictionary_to_vector(params):
    """
    Flattens parameters (W1,b1,W2,b2,...) into a column vector and returns the vector and list of keys for mapping.
    """
    keys = []
    vectors = []
    L = len(params) // 2
    for l in range(1, L + 1):
        W = params["W" + str(l)].reshape(-1, 1)
        b = params["b" + str(l)].reshape(-1, 1)
        vectors.append(W); vectors.append(b)
        keys.extend(["W" + str(l)] * W.shape[0])
        keys.extend(["b" + str(l)] * b.shape[0])
    return np.vstack(vectors), keys

def vector_to_dictionary(theta_vec, template_params):
    """
    Reconstruct parameters dict from a flattened theta vector and template_params for shapes.
    """
    params = {}
    L = len(template_params) // 2
    idx = 0
    for l in range(1, L + 1):
        W_shape = template_params["W" + str(l)].shape
        b_shape = template_params["b" + str(l)].shape
        size_W = W_shape[0] * W_shape[1]
        size_b = b_shape[0] * b_shape[1]
        params["W" + str(l)] = theta_vec[idx:idx+size_W].reshape(W_shape); idx += size_W
        params["b" + str(l)] = theta_vec[idx:idx+size_b].reshape(b_shape); idx += size_b
    return params

def gradients_to_vector(grads, template_params):
    """
    Stacks gradients dW1, db1, dW2, db2, ... into a single column vector.
    """
    vectors = []
    L = len(template_params) // 2
    for l in range(1, L + 1):
        vectors.append(grads["dW" + str(l)].reshape(-1, 1))
        vectors.append(grads["db" + str(l)].reshape(-1, 1))
    return np.vstack(vectors)
