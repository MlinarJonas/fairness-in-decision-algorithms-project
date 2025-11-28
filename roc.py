import numpy as np
import matplotlib.pyplot as plt
from data_processing import get_data
from data_split import train_val_test_split
from logistic_regression import LogisticRegression
from utils import scores

def prepare_data():

    # get data
    df = get_data()

    # Split data
    train, val, test = train_val_test_split(df)

    # Prepare training sets
    X_train = train.drop(columns=['pass_bar']).to_numpy()
    y_train = train['pass_bar'].to_numpy()

    # Prepare validation sets
    X_val = val.drop(columns=['pass_bar']).to_numpy()
    y_val = val['pass_bar'].to_numpy()
    gender_val = val['gender'].to_numpy()

    return (X_train, y_train), (X_val, y_val, gender_val), test

def train_logistic(X_train, y_train):
    # Create and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def roc_by_group(y_true, y_score, group, thresholds):

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

def get_roc_points(threshold_limit=1000):

    # Prepare data
    (X_train, y_train), (X_val, y_val, gender_val), _ = prepare_data()
    
    # Train model
    model = train_logistic(X_train, y_train)

    # get scores once
    y_score_val = model.predict_proba(X_val)

    # Define thresholds
    thresholds = np.linspace(0, 1, threshold_limit)

    # Get ROC points by group
    fpr_dict, tpr_dict = roc_by_group(
        y_true=y_val,
        y_score=y_score_val,
        group=gender_val,
        thresholds=thresholds,
    )

    # Convert to list format
    fpr_groups = [fpr_dict[0], fpr_dict[1]]
    tpr_groups = [tpr_dict[0], tpr_dict[1]]

    return fpr_groups, tpr_groups

# Obtain ROC points
roc_points_gender = get_roc_points()









