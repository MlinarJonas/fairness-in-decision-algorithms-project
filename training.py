import numpy as np
from utils import sigmoid
from models import logistic_regression_weighted, reg_logistic_regression_weighted, initialize_weights_3layer


def train_logistic_regression_weighted(x_train, y_train, gamma=0.1, max_iters=2000, pos_weight_scale=1.0):
    """
    Train weighted binary logistic regression (no regularization).

    Args:
        x_train : np.ndarray
            Training feature matrix (including intercept column).
        y_train : np.ndarray
            Training target vector (0/1).
        gamma : float, optional
            Learning rate for gradient descent (default 0.5).
        max_iters : int, optional
            Number of gradient descent iterations (default 2000).
        pos_weight_scale : float, optional
            Scaling factor for weighting the positive class (default 1.0).

    Returns:
        w : np.ndarray
            Final trained weight vector.
        loss : float
            Final weighted logistic loss.
    """
    # Initialize weights
    initial_w = np.zeros(x_train.shape[1])

    # Train weighted logistic regression
    w, loss = logistic_regression_weighted(
        y=y_train,
        tx=x_train,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
        pos_weight_scale=pos_weight_scale
    )

    return w, loss

def train_reg_logistic_regression_weighted(x_train, y_train, lambda_=1e-6, gamma=0.5, max_iters=2000):
    """
    Train weighted L2-regularized logistic regression.

    Args:
        x_train : np.ndarray
            Training feature matrix (including intercept column).
        y_train : np.ndarray
            Training target vector (0/1).
        lambda_ : float, optional
            Regularization parameter (default 1e-6).
        gamma : float, optional
            Learning rate (default 0.5).
        max_iters : int, optional
            Number of gradient descent iterations (default 2000).

    Returns:
        w : np.ndarray
            Final trained weight vector.
        loss : float
            Final weighted regularized logistic loss.
    """
    # Initialize weights
    initial_w = np.zeros(x_train.shape[1])

    # Train model
    w, loss = reg_logistic_regression_weighted(
        y=y_train,
        tx=x_train,
        lambda_=lambda_,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma
    )

    return w, loss

def train_weighted_three_layer_nn(X, y, n_hidden1=64, n_hidden2=32, lr=0.01, lambda_=1e-4,
                                  epochs=800, verbose=True, class_weight=True):
    """
    3-layer NN with 2 hidden layers, class weighting, L2 regularization, and F1 tracking.
    """
    n_samples, n_features = X.shape
    W1, b1, W2, b2, W3, b3 = initialize_weights_3layer(n_features, n_hidden1, n_hidden2, 1)
    y = y.reshape(-1, 1)

    # --- Class weights ---
    if class_weight:
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        w_pos = n_samples / (2 * pos)
        w_neg = n_samples / (2 * neg)
    else:
        w_pos = w_neg = 1.0

    for epoch in range(epochs):
        # ---- Forward ----
        Z1 = X @ W1 + b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ W2 + b2
        A2 = np.tanh(Z2)
        Z3 = A2 @ W3 + b3
        A3 = sigmoid(Z3)

        # ---- Weighted Loss ----
        eps = 1e-15
        weight_matrix = np.where(y == 1, w_pos, w_neg)
        loss = -np.mean(weight_matrix * (y * np.log(A3 + eps) + (1 - y) * np.log(1 - A3 + eps)))
        loss += lambda_ * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) / 2

        # ---- Backward ----
        dZ3 = weight_matrix * (A3 - y)
        dW3 = (A2.T @ dZ3) / n_samples + lambda_ * W3
        db3 = np.mean(dZ3, axis=0, keepdims=True)
        
        dA2 = dZ3 @ W3.T
        dZ2 = dA2 * (1 - np.tanh(Z2)**2)
        dW2 = (A1.T @ dZ2) / n_samples + lambda_ * W2
        db2 = np.mean(dZ2, axis=0, keepdims=True)
        
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (1 - np.tanh(Z1)**2)
        dW1 = (X.T @ dZ1) / n_samples + lambda_ * W1
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # ---- Update ----
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3

        # ---- Compute F1 ----
        preds = (A3 >= 0.5).astype(int)
        TP = np.sum((preds == 1) & (y == 1))
        FP = np.sum((preds == 1) & (y == 0))
        FN = np.sum((preds == 0) & (y == 1))
        precision = TP / (TP + FP + 1e-15)
        recall = TP / (TP + FN + 1e-15)
        f1 = 2 * precision * recall / (precision + recall + 1e-15)

        # ---- Print ----
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | F1: {f1:.4f}")

    return (W1, b1, W2, b2, W3, b3)