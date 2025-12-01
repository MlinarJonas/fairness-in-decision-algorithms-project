import numpy as np
import matplotlib.pyplot as plt
from data_processing import get_data
from data_split import train_val_test_split
from logistic_regression import LogisticRegression
from utils import scores


def roc_by_group(y_true, y_score, group):

    # Define thresholds
    thresholds = np.linspace(0, 1, 1000)

    # Initialize dictionaries to hold FPR and TPR for each group
    unique_groups = np.unique(group)
    fpr_dict = {g: [] for g in unique_groups}
    tpr_dict = {g: [] for g in unique_groups}

    for thr in thresholds:

        # Predict labels based on threshold
        y_pred = (y_score >= thr).astype(int)

        # Calculate FPR and TPR for each group
        for g in unique_groups:
            mask = (group == g)
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]

            # Calculate scores
            _, _, recall_g, fpr_g, _ = scores(y_pred_g, y_true_g)
            tpr_dict[g].append(recall_g)
            fpr_dict[g].append(fpr_g)

    return fpr_dict, tpr_dict











