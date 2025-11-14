import numpy as np
from utils import sigmoid

def logistic_regression_weighted(y, tx, initial_w, max_iters, gamma, pos_weight_scale=1.0):
    """
    Performs binary logistic regression with weighting for class imbalance.

    Args:
        y: numpy array of shape (N,), values 0 or 1
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D,)
        max_iters: int, number of iterations
        gamma: float, step size
        pos_weight_scale: float, scale factor for weighting positive class

    Returns:
        w: final weights (numpy array of shape (D,))
        loss: final weighted logistic loss (float)
    """
    w = initial_w.copy()

    # Compute class weights inversely proportional to class frequency
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    ratio = n_neg / max(1, n_pos)

    weight_pos = ratio * pos_weight_scale
    weight_neg = 1.0
    weights = np.where(y == 1, weight_pos, weight_neg)

    # Gradient descent
    for i in range(max_iters):
        pred = sigmoid(tx @ w)
        grad = tx.T @ (weights * (pred - y)) / np.sum(weights)  # weighted gradient
        w -= gamma * grad

    # Final weighted logistic loss
    epsilon = 1e-15
    pred = sigmoid(tx @ w)
    loss = -np.mean(weights * (y * np.log(pred + epsilon) + (1 - y) * np.log(1 - pred + epsilon)))

    return w, loss

def reg_logistic_regression_weighted(y, tx, lambda_, initial_w, max_iters, gamma, pos_weight_scale=1.0):
    """
    Performs L2-regularized binary logistic regression with weighting for class imbalance.

    Args:
        y : np.ndarray of shape (N,)
            Binary target vector with values 0 or 1.
        tx : np.ndarray of shape (N, D)
            Feature matrix where each row is a sample and each column is a feature.
        lambda_ : float
            L2 regularization parameter (ridge penalty). Regularization is applied to all weights except the intercept.
        initial_w : np.ndarray of shape (D,)
            Initial weights for gradient descent.
        max_iters : int
            Number of iterations for gradient descent.
        gamma : float
            Step size (learning rate) for gradient descent.
        pos_weight_scale : float, optional (default=1.0)
            Scaling factor for weighting the positive class to handle class imbalance.
            Effective weight for positive class = (n_neg / n_pos) * pos_weight_scale.

    Returns:
        w : np.ndarray of shape (D,)
            Final weight vector after gradient descent.
        loss : float
            Final weighted regularized logistic loss.
            Computed as the negative log-likelihood weighted by class weights,
            plus the L2 penalty on all weights except the intercept.
    """
    w = initial_w.copy()

    # Compute class weights inversely proportional to class frequency
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    ratio = n_neg / max(1, n_pos)

    weight_pos = ratio * pos_weight_scale
    weight_neg = 1.0
    weights = np.where(y == 1, weight_pos, weight_neg)

    for i in range(max_iters):
        pred = sigmoid(tx @ w)
        # Gradient (no regularization on intercept)
        grad = tx.T @ (weights * (pred - y)) / np.sum(weights) #y.size
        grad[1:] += 2 * lambda_ * w[1:]
        w -= gamma * grad

    # Final loss (no reg on intercept)
    epsilon = 1e-15
    pred = sigmoid(tx @ w)
    loss = (
        -np.mean(weights * (y * np.log(pred + epsilon) + (1 - y) * np.log(1 - pred + epsilon)))
        + lambda_ * np.sum(w[1:] ** 2)
    )

    return w, loss

def initialize_weights_3layer(n_in, n_hidden1, n_hidden2, n_out):
    # Xavier initialization
    W1 = np.random.randn(n_in, n_hidden1) * np.sqrt(2 / (n_in + n_hidden1))
    b1 = np.zeros((1, n_hidden1))
    W2 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(2 / (n_hidden1 + n_hidden2))
    b2 = np.zeros((1, n_hidden2))
    W3 = np.random.randn(n_hidden2, n_out) * np.sqrt(2 / (n_hidden2 + n_out))
    b3 = np.zeros((1, n_out))
    return W1, b1, W2, b2, W3, b3


def predict_three_layer_nn(X, params, threshold=0.5):
    W1, b1, W2, b2, W3, b3 = params
    A1 = np.tanh(X @ W1 + b1)
    A2 = np.tanh(A1 @ W2 + b2)
    probs = sigmoid(A2 @ W3 + b3).flatten()
    preds = (probs >= threshold).astype(int)
    return preds, probs


def ensemble_predictions(X, params_nn, w_logreg, weight_nn=0.3, weight_logreg=0.7, threshold=0.5):
    """
    Ensemble predictions from a 3-layer NN and a logistic regression model.
    
    Args:
        X: np.ndarray — input features (same preprocessing as training)
        params_nn: tuple — (W1, b1, W2, b2, W3, b3) from train_weighted_three_layer_nn
        w_logreg: np.ndarray — trained logistic regression weights
        weight_nn: float — contribution weight for NN probabilities
        weight_logreg: float — contribution weight for logistic regression
        threshold: float — probability threshold for binary classification (default 0.5)
    
    Returns:
        y_pred_final: np.ndarray — final binary predictions (0/1)
        y_prob_final: np.ndarray — combined probabilities (float)
        y_submission: np.ndarray — mapped predictions (-1/1 for competition)
    """
    
    # --- Neural network forward pass ---
    W1, b1, W2, b2, W3, b3 = params_nn
    A1 = np.tanh(X @ W1 + b1)
    A2 = np.tanh(A1 @ W2 + b2)
    y_prob_nn = 1 / (1 + np.exp(-(A2 @ W3 + b3)))  # sigmoid
    y_prob_nn = y_prob_nn.flatten()

    # --- Logistic regression probabilities ---
    y_prob_logreg = 1 / (1 + np.exp(-(X @ w_logreg)))
    y_prob_logreg = y_prob_logreg.flatten()

    # --- Weighted average of probabilities ---
    y_prob_final = weight_nn * y_prob_nn + weight_logreg * y_prob_logreg

    # --- Binary predictions ---
    y_pred_final = (y_prob_final >= threshold).astype(int)

    # --- Map to -1/1 for submission ---
    y_submission = np.where(y_pred_final == 0, -1, 1)

    return y_pred_final, y_prob_final, y_submission

