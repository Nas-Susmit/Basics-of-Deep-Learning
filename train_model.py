# train_model.py
import numpy as np
from forward_backward import L_model_forward, L_model_backward
from optimizers import (update_parameters_with_gd,
                        initialize_velocity, update_parameters_with_momentum,
                        initialize_adam, update_parameters_with_adam,
                        update_lr, schedule_lr_decay)
from utils import initialize_parameters_he
from random import seed as random_seed
from split_data import train_dev_test_split
from data_preprocessing import make_dataset
from visualize import plot_decision_boundary
from metrics import evaluate

def model(X, Y, layers_dims, optimizer="adam", learning_rate=0.01, mini_batch_size=64,
          num_epochs=1000, lambd=0.0, keep_probs=None,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
          lr_decay=None, lr_decay_rate=0.0, lr_time_interval=1000,
          print_cost=True, seed_val=10):
    """
    Training driver. Returns parameters.
    """
    np.random.seed(3)
    parameters = initialize_parameters_he(layers_dims)
    costs = []
    t = 0
    seed_val_local = seed_val
    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    m = X.shape[1]
    for epoch in range(num_epochs):
        seed_val_local += 1
        # create minibatches
        from forward_backward import random_mini_batches as rmmb if False else None  # placeholder
        # we will use our own minibatcher to avoid import tie-ups - define inline small helper:
        permutation = list(np.random.permutation(m))
        X_shuf = X[:, permutation]
        Y_shuf = Y[:, permutation].reshape(1, m)

        num_complete = m // mini_batch_size
        minibatches = []
        for k in range(num_complete):
            mini_X = X_shuf[:, k*mini_batch_size:(k+1)*mini_batch_size]
            mini_Y = Y_shuf[:, k*mini_batch_size:(k+1)*mini_batch_size]
            minibatches.append((mini_X, mini_Y))
        if m % mini_batch_size != 0:
            mini_X = X_shuf[:, num_complete*mini_batch_size:m]
            mini_Y = Y_shuf[:, num_complete*mini_batch_size:m]
            minibatches.append((mini_X, mini_Y))

        # learning rate schedule
        if lr_decay == 'inverse':
            lr = update_lr(learning_rate, epoch, lr_decay_rate)
        elif lr_decay == 'schedule':
            lr = schedule_lr_decay(learning_rate, epoch, lr_decay_rate, time_interval=lr_time_interval)
        else:
            lr = learning_rate

        cost_sum = 0.0
        for mini_X, mini_Y in minibatches:
            AL, caches = L_model_forward(mini_X, parameters, keep_probs=keep_probs, seed=epoch)
            cost_sum += (lambda: 0)()  # placeholder
            # compute cost inline
            m_batch = mini_Y.shape[1]
            ALc = np.clip(AL, 1e-12, 1 - 1e-12)
            cost = - (1.0 / m_batch) * np.sum(mini_Y * np.log(ALc) + (1-mini_Y) * np.log(1-ALc))
            if lambd > 0:
                L2 = 0.0
                for l in range(1, len(parameters)//2 + 1):
                    L2 += np.sum(parameters["W" + str(l)]**2)
                cost += (lambd / (2.0 * m_batch)) * L2
            cost_sum += cost

            grads = L_model_backward(AL, mini_Y, caches, lambd=lambd)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, lr)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=beta, lr=lr)
            elif optimizer == "adam":
                t += 1
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s, t,
                                                                     lr=lr, beta1=beta1, beta2=beta2, eps=epsilon)
            else:
                raise ValueError("Unknown optimizer")

        avg_cost = cost_sum / len(minibatches)
        if print_cost and epoch % 100 == 0:
            print(f"Cost after epoch {epoch}: {avg_cost:.6f}")
            costs.append(avg_cost)

    # learning curve plotted by visualize module if desired by caller
    return parameters
