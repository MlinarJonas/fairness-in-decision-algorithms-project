import numpy as np
from utils import sigmoid

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