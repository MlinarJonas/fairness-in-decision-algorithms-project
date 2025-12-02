import numpy as np
from utils.utils import scores


def compute_roc_points(y_true, y_score):

    thresholds = np.linspace(0, 1, 1000)
    
    fpr_list = []
    tpr_list = []

    for thr in thresholds:

        # Predict labels based on threshold
        y_pred = (y_score >= thr).astype(int)

        # Calculate FPR and TPR
        _, _, tpr, fpr, _ = scores(y_pred, y_true)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return thresholds, np.array(fpr_list), np.array(tpr_list)

def compute_roc_points_by_group(y_true, y_score, group):

    thresholds = np.linspace(0, 1, 1000)
    
    # Initialize dictionaries to hold FPR and TPR for each group
    unique_groups = np.unique(group)
    roc_points_by_group = {g: {'fpr': [], 'tpr' : []} for g in unique_groups}  # g: (fpr_list, tpr_list)
    
    for thr in thresholds:

        # Predict labels based on threshold
        y_pred = (y_score >= thr).astype(int)

        # Calculate FPR and TPR for each group
        for g in unique_groups:
            mask = (group == g)
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]

            # Calculate scores
            _, _, tpr_g, fpr_g, _ = scores(y_pred_g, y_true_g)
            roc_points_by_group[g]['fpr'].append(fpr_g)
            roc_points_by_group[g]['tpr'].append(tpr_g)
    
    # convert to arrays
    for g in unique_groups:
        roc_points_by_group[g]['fpr'] = np.array(roc_points_by_group[g]['fpr'])
        roc_points_by_group[g]['tpr'] = np.array(roc_points_by_group[g]['tpr'])

    return thresholds, roc_points_by_group





