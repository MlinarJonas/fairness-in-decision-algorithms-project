import numpy as np
from utils import build_k_indices, sigmoid, scores
from training import train_reg_logistic_regression_weighted
from logistic_regression import reg_logistic_regression_weighted

def cross_validate_model(y, X, model_func, k_fold=5, seed=42):
    """
    Perform k-fold cross-validation using a model function.
    
    Parameters
    ----------
    y : np.ndarray
        Target vector (0/1).
    X : np.ndarray
        Feature matrix.
    model_func : callable
        Function with signature (X_train, y_train, X_test) → y_pred.
    k_fold : int
        Number of folds.
    seed : int
        Random seed.
        
    Returns
    -------
    metrics : dict
        Dict containing lists of false positive, true positive, accuracy, precision, recall, and F1 scores for each fold.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    fps, tps, accs, precs, recs, f1s = [],[],[], [], [], []

    for k in range(k_fold):
        test_idx = k_indices[k]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_idx], y[test_idx]

        y_pred = model_func(X_train, y_train, X_test)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        fp, tp, acc, prec, rec, f1 = scores(y_pred, y_test)

        fps.append(fp)
        tps.append(tp)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    return {
        "false positive": np.array(fps),
        "true positive": np.array(tps),
        "accuracy": np.array(accs),
        "precision": np.array(precs),
        "recall": np.array(recs),
        "f1": np.array(f1s)
    }

def reg_weighted_lr_model(X_train, y_train, X_test):
    w, _ = train_reg_logistic_regression_weighted(X_train, y_train, lambda_=1e-6, gamma=0.5, max_iters=2000)
    y_pred = (sigmoid(X_test @ w) >= 0.5).astype(int)
    return y_pred