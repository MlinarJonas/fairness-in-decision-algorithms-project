import numpy as np
from utils import sigmoid

class LogisticRegression:
    def __init__(self, lambda_=1e-6, max_iters=2000, gamma=0.5):
        self.lambda_ = lambda_
        self.max_iters = max_iters
        self.gamma = gamma
        self.loss = None
        self.w = None

    def fit(self, tx, y):
        self.w, self.loss = reg_logistic_regression_weighted(
            y=y,
            tx=tx,
            lambda_=self.lambda_,
            max_iters=self.max_iters,
            gamma=self.gamma
        )

    def predict_proba(self, tx):
        return sigmoid(tx @ self.w)

    def predict(self, tx, threshold=0.5):
        proba = self.predict_proba(tx)
        return (proba >= threshold).astype(int)

def reg_logistic_regression_weighted(tx, y, lambda_, max_iters, gamma, pos_weight_scale=1.0):

    # Initialize weights
   
    w = np.zeros(tx.shape[1])
    
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