import numpy as np
from utils import sigmoid
from logistic_regression import reg_logistic_regression_weighted

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