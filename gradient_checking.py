# gradient_checking.py
import numpy as np
from utils import dictionary_to_vector, vector_to_dictionary, gradients_to_vector
from forward_backward import L_model_forward, L_model_backward
from utils import initialize_parameters_he

def forward_for_gradcheck(X, Y, params, lambd=0.0, keep_probs=None):
    AL, caches = L_model_forward(X, params, keep_probs=keep_probs)
    from forward_backward import compute_cost if False else None  # no-op placeholder (compute_cost used below)
    # instead compute cost inline to avoid circular imports
    m = Y.shape[1]
    AL_clipped = np.clip(AL, 1e-12, 1 - 1e-12)
    cost = - (1.0 / m) * np.sum(Y * np.log(AL_clipped) + (1 - Y) * np.log(1 - AL_clipped))
    # add L2 if lambda provided
    if lambd > 0.0:
        L2 = 0.0
        for l in range(1, len(params)//2 + 1):
            L2 += np.sum(np.square(params["W" + str(l)]))
        cost += (lambd / (2.0 * m)) * L2
    return cost, AL, caches

def gradient_check(parameters, X, Y, lambd=0.0, epsilon=1e-7, keep_probs=None, print_msg=False):
    """
    Vectorized gradient check across all parameters.
    """
    theta, _ = dictionary_to_vector(parameters)
    num_params = theta.shape[0]
    J_plus  = np.zeros((num_params, 1))
    J_minus = np.zeros((num_params, 1))
    gradapprox = np.zeros((num_params, 1))

    # Backprop gradients
    _, AL, caches = forward_for_gradcheck(X, Y, parameters, lambd=lambd, keep_probs=keep_probs)
    grads = L_model_backward(AL, Y, caches, lambd=lambd)
    grad = gradients_to_vector(grads, parameters)

    for i in range(num_params):
        theta_plus = np.copy(theta);  theta_plus[i, 0] += epsilon
        theta_minus = np.copy(theta); theta_minus[i, 0] -= epsilon

        params_plus  = vector_to_dictionary(theta_plus, parameters)
        params_minus = vector_to_dictionary(theta_minus, parameters)

        J_plus[i, 0], _, _  = forward_for_gradcheck(X, Y, params_plus,  lambd=lambd, keep_probs=keep_probs)
        J_minus[i, 0], _, _ = forward_for_gradcheck(X, Y, params_minus, lambd=lambd, keep_probs=keep_probs)

        gradapprox[i, 0] = (J_plus[i, 0] - J_minus[i, 0]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox) + 1e-12
    diff = numerator / denominator
    if print_msg:
        if diff > 2e-7:
            print("\033[93mGradient check FAILED: difference = {:.3e}\033[0m".format(diff))
        else:
            print("\033[92mGradient check PASSED: difference = {:.3e}\033[0m".format(diff))
    return diff
