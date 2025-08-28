# main.py
from data_preprocessing import make_dataset
from train_model import model
from utils import initialize_parameters_he
from gradient_checking import gradient_check
from visualize import plot_decision_boundary, plot_learning_curve
from metrics import evaluate, predict
from forward_backward import L_model_forward

if __name__ == "__main__":
    # 1) Data
    X, Y, scaler = make_dataset(n_samples=1000, noise=0.2)

    # 2) Split manually (simple): train/dev/test (80/10/10)
    from split_data import train_dev_test_split
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = train_dev_test_split(X, Y, test_size=0.1, dev_size=0.1)

    print("Shapes:", X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape, X_test.shape, Y_test.shape)

    # 3) Model config
    layers = [X_train.shape[0], 20, 7, 5, 1]

    # 4) Quick gradient check on very small subset (disable dropout)
    params_gc = initialize_parameters_he(layers)
    X_gc = X_train[:, :5]
    Y_gc = Y_train[:, :5]
    diff = gradient_check(params_gc, X_gc, Y_gc, lambd=0.0, keep_probs=[1.0, 1.0, 1.0], print_msg=True)

    # 5) Train
    params = model(X_train, Y_train, layers_dims=layers,
                   optimizer="adam", learning_rate=0.01, mini_batch_size=64,
                   num_epochs=1000, lambd=0.7, keep_probs=[0.9, 0.9, 1.0],
                   lr_decay='schedule', lr_decay_rate=0.05, lr_time_interval=250,
                   print_cost=True)

    # 6) Evaluate
    from forward_backward import L_model_forward
    from metrics import evaluate as ev
    ev(params, X_train, Y_train, forward_func=L_model_forward, title="Train")
    ev(params, X_dev, Y_dev, forward_func=L_model_forward, title="Dev")
    ev(params, X_test, Y_test, forward_func=L_model_forward, title="Test")

    # 7) Decision boundary
    plot_decision_boundary(lambda x: predict(params, x, forward_func=L_model_forward), X_train, Y_train,
                           title="Decision Boundary (Train)")
