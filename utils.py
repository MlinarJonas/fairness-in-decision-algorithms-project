import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def scores(y_pred, y_true):
    """
    Compute classification metrics from predicted and true binary labels.

    Parameters
    ----------
    y_pred : array-like
        Predicted binary labels (0 or 1).
    y_true : array-like
        True binary labels (0 or 1).

    Returns
    -------
    accuracy : float
        (TP + TN) / total samples
    precision : float
        TP / (TP + FP)
    recall : float
        TP / (TP + FN)
    fpr : float
        FP / (FP + TN), false positive rate
    f1 : float
        Harmonic mean of precision and recall
    """
    
    # Flatten arrays
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    # Calculate TP, TN, FP, FN
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, fpr, f1

def build_k_indices(y, k_fold, seed):
    np.random.seed(seed)
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

