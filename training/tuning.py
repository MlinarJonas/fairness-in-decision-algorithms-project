"""
Module for hyperparameter tuning using cross-validation.
Contains functions for searching optimal parameters for logistic regression models
(regularized or not) based on F1-score.
"""
import numpy as np
from utils.utils import sigmoid, f1_score, build_k_indices
from models.logistic_regression import reg_logistic_regression_weighted

def cross_validation_tuning(y, X, k_indices, gammas, pos_weight_scales, model_func,
                            lambdas=None, max_iters=2000):
    """
    Perform K-fold cross-validation to tune hyperparameters for weighted logistic regression models.

    Args:
        y (np.ndarray): Binary target vector (0/1).
        X (np.ndarray): Feature matrix.
        k_indices (list of np.ndarray): Indices for each fold in K-fold CV.
        gammas (list of float): Learning rates to test.
        pos_weight_scales (list of float): Scaling factors for positive class weight.
        model_func (callable): Training function to call
            (e.g., reg_logistic_regression_weighted or logistic_regression_weighted).
        lambdas (list of float or None): Regularization strengths (ignored if model_func has no lambda).
        max_iters (int): Maximum number of iterations per training.

    Returns:
        best_params (dict): Dictionary containing the best hyperparameters.
        best_f1 (float): Best mean F1-score obtained across folds.
    """
    best_f1 = 0
    best_params = {}

    # Dummy list if no regularization
    if lambdas is None:
        lambdas = [0.0]

    for pos_scale in pos_weight_scales:
        for lambda_ in lambdas:
            for gamma in gammas:
                f1_scores = []

                # Cross-validation loop
                for k in range(len(k_indices)):
                    te_idx = k_indices[k]
                    tr_idx = np.hstack([k_indices[i] for i in range(len(k_indices)) if i != k])
                    X_tr, y_tr = X[tr_idx], y[tr_idx]
                    X_te, y_te = X[te_idx], y[te_idx]

                    initial_w = np.zeros(X.shape[1])

                    # Handle regularized / non-regularized functions automatically
                    if "lambda_" in model_func.__code__.co_varnames:
                        w, _ = model_func(y_tr, X_tr, lambda_, initial_w, max_iters, gamma, pos_scale)
                    else:
                        w, _ = model_func(y_tr, X_tr, initial_w, max_iters, gamma, pos_scale)

                    # Predictions
                    y_pred_prob = sigmoid(X_te @ w)
                    y_pred = (y_pred_prob >= 0.5).astype(int)

                    # Evaluate
                    f1 = f1_score(y_te, y_pred)
                    f1_scores.append(f1)

                mean_f1 = np.mean(f1_scores)
                print(f"scale={pos_scale}, λ={lambda_:.1e}, γ={gamma:.1e} → mean F1={mean_f1:.4f}")

                if mean_f1 > best_f1:
                    best_f1 = mean_f1
                    best_params = {
                        "lambda_": lambda_,
                        "gamma": gamma,
                        "pos_weight_scale": pos_scale
                    }

    print("\nBest parameters:", best_params, f"→ F1={best_f1:.4f}")
    return best_params, best_f1


