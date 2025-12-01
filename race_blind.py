import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from data_processing import get_data
from data_split import train_val_test_split
from logistic_regression import LogisticRegression
from utils import scores

def get_roc_points_one_curve(threshold_limit=1000):
    # Load and preprocess data
    df = get_data()

    # Split data into train, validation, and test sets
    train, val, test = train_val_test_split(df)

    X_train = train.drop(columns=['pass_bar']).to_numpy()
    y_train = train['pass_bar'].to_numpy()

    X_val = val.drop(columns=['pass_bar']).to_numpy()
    y_val = val['pass_bar'].to_numpy()

    # thresholds for ROC curve
    thresholds = np.linspace(0, 1, threshold_limit)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Prepare lists for overall ROC
    tprs, fprs = [], []

    for threshold in thresholds:

        # Predict on the whole validation set
        y_pred_val = model.predict(X_val, threshold=threshold)

        # Compute FPR, TPR for all individuals together
        _, _, recall, fpr, _ = scores(y_pred_val, y_val)

        tprs.append(recall)
        fprs.append(fpr)

    return np.array(fprs), np.array(tprs)



def find_optimal_gamma(roc_points, l01 = 1, l10 = 1):
    """
    roc_points: list of tuples (FPR, TPR) describing the ROC curve.
                They do NOT have to be sorted.
                
    loss: function taking a pair (y_true, y_pred) in {(1,0),(0,1)} and returning l(y_true, y_pred)
          Example:
              def loss(pair):
                  if pair == (1,0): return c_fp
                  if pair == (0,1): return c_fn
                  
    Returns:
        gamma_star: optimal point (gamma_0, gamma_1)
        value: the minimal loss value
        index: index of the optimal ROC point in the sorted list
    """

    # Sort ROC points by FPR (conventional ROC ordering)
    roc = sorted(roc_points, key=lambda p: p[0])

    best_value = float('inf')
    best_gamma = None
    best_idx = None

    for i, (fpr, tpr) in enumerate(roc):
        fnr = 1 - tpr
        value = fpr * l10 + fnr * l01

        if value < best_value:
            best_value = value
            best_gamma = (fpr, fnr)
            best_idx = i

    return best_gamma, best_value, best_idx
