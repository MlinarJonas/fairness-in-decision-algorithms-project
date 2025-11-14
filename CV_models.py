import numpy as np
from utils import build_k_indices, sigmoid
from training import train_logistic_regression_weighted, train_reg_logistic_regression_weighted, train_weighted_three_layer_nn
from models import predict_three_layer_nn
from models import ensemble_predictions


def scores(y_pred, y_true):
    """
    Compute evaluation metrics: accuracy, precision, recall, and F1-score.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).
    y_true : np.ndarray
        True binary labels (0 or 1).
        
    Returns
    -------
    accuracy : float
    precision : float
    recall : float
    f1 : float
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return accuracy, precision, recall, f1

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
        Dict containing lists of accuracy, precision, recall, and F1 scores for each fold.
    """
    k_indices = build_k_indices(y, k_fold, seed)
    accs, precs, recs, f1s = [], [], [], []

    for k in range(k_fold):
        test_idx = k_indices[k]
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_idx], y[test_idx]

        y_pred = model_func(X_train, y_train, X_test)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        acc, prec, rec, f1 = scores(y_pred, y_test)

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

    return {
        "accuracy": np.array(accs),
        "precision": np.array(precs),
        "recall": np.array(recs),
        "f1": np.array(f1s)
    }

def weighted_lr_model(X_train, y_train, X_test):
    w, _ = train_logistic_regression_weighted(X_train, y_train, gamma=0.5, max_iters=2000, pos_weight_scale=1.0)
    y_pred = (sigmoid(X_test @ w) >= 0.5).astype(int)
    return y_pred

def reg_weighted_lr_model(X_train, y_train, X_test):
    w, _ = train_reg_logistic_regression_weighted(X_train, y_train, lambda_=1e-6, gamma=0.5, max_iters=2000)
    y_pred = (sigmoid(X_test @ w) >= 0.5).astype(int)
    return y_pred

def nn_weighted_model(X_train, y_train, X_test):
    params_3 = train_weighted_three_layer_nn(X_train, y_train,
                                             n_hidden1=128, n_hidden2=64,
                                             lr=0.01, lambda_=1e-4, epochs=500, class_weight=True)
    return predict_three_layer_nn(X_test, params_3)
      
    

def ensemble_model(X_train, y_train, X_test):
    from training import train_reg_logistic_regression_weighted, train_weighted_three_layer_nn
    from models import ensemble_predictions
    import numpy as np

    # Train logistic regression
    w, _ = train_reg_logistic_regression_weighted(
        X_train, y_train, lambda_=1e-6, gamma=0.5, max_iters=2000
    )

    # Train neural network
    params_3 = train_weighted_three_layer_nn(
        X_train, y_train,
        n_hidden1=128, n_hidden2=64,
        lr=0.01, lambda_=1e-4, epochs=500, class_weight=True
    )

    # Ensemble both models
    _, _, y_submission = ensemble_predictions(
        X_test, params_3, w,
        weight_nn=0.3, weight_logreg=0.7
    )

    # Convert -1/1 back to 0/1 for F1 and accuracy
    y_pred = (y_submission == 1).astype(int)
    return y_pred

