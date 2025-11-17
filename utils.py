import numpy as np

def sigmoid(z):
    """Compute the sigmoid function"""
    return 1 / (1 + np.exp(-z))

def f1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    if TP + FP == 0 or TP + FN == 0:
        return 0.0  # avoid division by zero
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def scores(y_pred, y_true):
    """
    Compute evaluation metrics: false positive, true positive, accuracy, precision, recall, and F1-score.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).
    y_true : np.ndarray
        True binary labels (0 or 1).
        
    Returns
    -------
    false positive : float
    true positive : float
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

    return fp, tp, accuracy, precision, recall, f1

def build_k_indices(y, k_fold, seed):
    np.random.seed(seed)
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
