import numpy as np 
import matplotlib.pyplot as plt
from metrics.threshold import find_threshold_on_single_roc
from data.data_processing import download_dataset, load_dataset, process_dataset, prepare_data_for_training
from training.train import train_logistic_regression
from metrics.roc import compute_roc_points_by_group, compute_roc_points
from utils.utils import Youden_J, Youden_J_groups
from utils.plots import plot_single_roc_curve, plot_grouped_roc_curves, plot_group_thresholds
from models.optimize_gamma import solve_gamma_from_roc_points_equal_odds, solve_gammas_from_roc_points_equal_opportunity, class_distribution_by_group, solve_gammas_from_roc_points_demographic_parity
from metrics.threshold import find_threshold_on_single_roc

def load_and_preprocess_data():
    path = download_dataset()
    df_raw = load_dataset(path)
    df_processed = process_dataset(df_raw)
    splits = prepare_data_for_training(df_processed)
    return splits, df_processed

def train_model(X_train, y_train):
    model = train_logistic_regression(X_train, y_train)
    return model
    
def evaluate_model(model, X_val, y_val, gender_val):
    
    # Get prediction scores
    y_score = model.predict_proba(X_val)
    
    # Compute ROC points overall
    thresholds, fpr, tpr = compute_roc_points(y_val, y_score)

    # Compute ROC points by group
    thresholds_by_group, roc_points_by_group = compute_roc_points_by_group(y_val, y_score, gender_val)
    groups = sorted(roc_points_by_group.keys())
    amount_groups = len(groups)
    fpr_groups = [roc_points_by_group[g]["fpr"] for g in groups]
    tpr_groups = [roc_points_by_group[g]["tpr"] for g in groups]

    # Get class distributions by group
    distribution = class_distribution_by_group(y_val, gender_val, thresholds_by_group)
    groups = sorted(distribution.keys()) # consistent group order
    pi0 = np.array([distribution[g]["joint"]["P(A=g,Y=0)"] for g in groups], dtype=float)  # pi0[a] ∝ P(A=a, Y=0)
    pi1 = np.array([distribution[g]["joint"]["P(A=g,Y=1)"] for g in groups], dtype=float)  # pi1[a] ∝ P(A=a, Y=1)

    # Compute optimal point and threshold --- Max Profit
    max_profit = Youden_J_groups(fpr_groups, tpr_groups, thresholds_by_group)
    optimal_points_max_profit = max_profit["optimal_point"]
    optimal_thresholds_max_profit = max_profit["optimal_threshold"]

    # Compute optimal point and threshold -- - Single Threshold
    optimal_threshold_single_threshold = [Youden_J(fpr, tpr, thresholds)[0]] * amount_groups

    # Compute optimal point and threshold --- Equal Odds
    results_equal_odds = solve_gamma_from_roc_points_equal_odds(fpr_groups, tpr_groups, thresholds_by_group)
    
    # Compute optimal point and threshold --- Equal Opportunity
    results_equal_opportunity = solve_gammas_from_roc_points_equal_opportunity(fpr_groups, tpr_groups, pi0, pi1,thresholds_by_group)

    # Compute optimal point and threshold --- Demographic Parity
    results_demographic_parity = solve_gammas_from_roc_points_demographic_parity(fpr_groups, tpr_groups, pi0, pi1, thresholds_by_group)


    evaluation_results = {
        "Overall": {
            "thresholds": thresholds,
            "fpr": fpr,
            "tpr": tpr,
            "optimal_threshold" : optimal_threshold_single_threshold,
            "optimal_point" : Youden_J(fpr, tpr, thresholds)[1]
        },
        "By_Group": {
            g : {
                "group": g,
                "thresholds": thresholds_by_group,
                "fpr": fpr_groups[g],
                "tpr": tpr_groups[g],
            } for g in roc_points_by_group
        },
        "Max_Profit": {
            "optimal_threshold": optimal_thresholds_max_profit,
            "optimal_point": optimal_points_max_profit
        },

        "Equal_Odds": {
            "optimal_threshold": results_equal_odds["optimal_threshold"],
            "optimal_point": results_equal_odds["optimal_point"]
        },
        "Equal_Opportunity": {
            "optimal_threshold": results_equal_opportunity["optimal_threshold"],
            "optimal_point": results_equal_opportunity["optimal_point"]
        },
        "Demographic_Parity": {
            "optimal_threshold": results_demographic_parity["optimal_threshold"],
            "optimal_point": results_demographic_parity["optimal_point"]
        }
    }
    return evaluation_results

def plot_results(evaluation_results):
    
    
    # Plot max profit ROC curve
    plot_grouped_roc_curves(
        roc_points = evaluation_results["By_Group"],
        labels_group = ["Male", "Female"],
        optimal_points = evaluation_results["Max_Profit"]["optimal_point"],
        fairness_label = "Max Profit"
    )

    # Plot single threshold ROC curve
    plot_single_roc_curve(
        fpr=evaluation_results["Overall"]["fpr"],
        tpr=evaluation_results["Overall"]["tpr"],
        optimal_point=evaluation_results["Overall"]["optimal_point"],
        label = "Roc Curve single threshold"
    )
    
    # Plot equal opportunity ROC curve
    plot_grouped_roc_curves(
        roc_points = evaluation_results["By_Group"],
        labels_group = ["Male", "Female"],
        optimal_points = evaluation_results["Equal_Opportunity"]["optimal_point"],
        fairness_label = "Equal Opportunity"
    )
    
    # Plot equal odds ROC curve
    plot_grouped_roc_curves(
        roc_points = evaluation_results["By_Group"],
        labels_group = ["Male", "Female"],
        optimal_points = evaluation_results["Equal_Odds"]["optimal_point"],
        fairness_label = "Equal Odds"
    )
    
    # Plot demographic parity ROC curve
    plot_grouped_roc_curves(
        roc_points = evaluation_results["By_Group"],
        labels_group = ["Male", "Female"],
        optimal_points = evaluation_results["Demographic_Parity"]["optimal_point"],
        fairness_label = "Demographic Parity"
    )
    

def run_fairness_analysis():
    # TODO implement fairness analysis
    pass


if __name__ == "__main__":
    splits, df = load_and_preprocess_data()
    (X_train, y_train), (X_val, y_val, gender_val), test = splits
    model = train_model(X_train, y_train)
    evaluation_results = evaluate_model(model, X_val, y_val, gender_val)
    print(evaluation_results["Max_Profit"]["optimal_threshold"])
    print(evaluation_results["Overall"]["optimal_threshold"])
    print(evaluation_results["Equal_Opportunity"]["optimal_threshold"])
    print(evaluation_results["Equal_Odds"]["optimal_threshold"])
    print(evaluation_results["Demographic_Parity"]["optimal_threshold"])
    #plot_results(evaluation_results)
    plot_group_thresholds(evaluation_results)
    #plot_group_thresholds_test(evaluation_results)