# optimizers.py
import numpy as np

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v

def update_parameters_with_gd(parameters, grads, lr):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= lr * grads["dW" + str(l)]
        parameters["b" + str(l)] -= lr * grads["db" + str(l)]
    return parameters

def update_parameters_with_momentum(parameters, grads, v, beta=0.9, lr=0.01):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        parameters["W" + str(l)] -= lr * v["dW" + str(l)]
        parameters["b" + str(l)] -= lr * v["db" + str(l)]
    return parameters, v

def initialize_adam(parameters):
    L = len(parameters) // 2
    v, s = {}, {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, lr=0.01,
                                beta1=0.9, beta2=0.999, eps=1e-8):
    L = len(parameters) // 2
    v_corr, s_corr = {}, {}
    for l in range(1, L + 1):
        # first moment
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        v_corr["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corr["db" + str(l)] = v["db" + str(l)] / (1 - beta1 ** t)

        # second moment
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])
        s_corr["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corr["db" + str(l)] = s["db" + str(l)] / (1 - beta2 ** t)

        # update
        parameters["W" + str(l)] -= lr * (v_corr["dW" + str(l)] / (np.sqrt(s_corr["dW" + str(l)]) + eps))
        parameters["b" + str(l)] -= lr * (v_corr["db" + str(l)] / (np.sqrt(s_corr["db" + str(l)]) + eps))
    return parameters, v, s, v_corr, s_corr

# Learning rate schedules
def update_lr(lr0, epoch_num, decay_rate):
    return lr0 / (1 + decay_rate * epoch_num)

def schedule_lr_decay(lr0, epoch_num, decay_rate, time_interval=1000):
    return lr0 / (1 + decay_rate * (epoch_num // time_interval))
