import numpy as np 
import matplotlib.pyplot as plt
from metrics.threshold import find_threshold_on_single_roc
from data.data_processing import download_dataset, load_dataset, process_dataset, prepare_data_for_training
from training.train import train_logistic_regression
from metrics.roc import compute_roc_points_by_group, compute_roc_points
from utils.utils import Youden_J
from utils.plots import plot_single_roc_curve, plot_grouped_roc_curves
from models.optimize_gamma import solve_gamma_from_roc_points_equal_odds, solve_gammas_from_roc_points_equal_opportunity, class_distribution_by_group, solve_gammas_from_roc_points_demographic_parity

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
    
    # Get class distributions by group
    distribution = class_distribution_by_group(y_val, gender_val)
    groups = sorted(distribution.keys()) # consistent group order
    pi0 = np.array([distribution[g]["joint"]["P(A=g,Y=0)"] for g in groups], dtype=float)  # pi0[a] ∝ P(A=a, Y=0)
    pi1 = np.array([distribution[g]["joint"]["P(A=g,Y=1)"] for g in groups], dtype=float)  # pi1[a] ∝ P(A=a, Y=1)

    # Compute ROC points overall
    thresholds, fpr, tpr = compute_roc_points(y_val, y_score)

    # Compute ROC points by group
    thresholds_by_group, roc_points_by_group = compute_roc_points_by_group(y_val, y_score, gender_val)

    # Consistent group order for everything
    groups = sorted(distribution.keys())

    # Compute optimal points and threshold --- Max Profit
    optimal_thresholds_max_profit = []
    optimal_point_max_profit = []

    for g in roc_points_by_group:
        best_threshold, point = Youden_J(
            roc_points_by_group[g]["fpr"],
            roc_points_by_group[g]["tpr"],
            thresholds_by_group
        )
        optimal_thresholds_max_profit.append(best_threshold)
        optimal_point_max_profit.append(point)

    # Compute optimal point and threshold --- Single Threshold 
    groups = sorted(roc_points_by_group.keys())
    fpr_groups = [roc_points_by_group[g]["fpr"] for g in groups]
    tpr_groups = [roc_points_by_group[g]["tpr"] for g in groups]
    optimal_threshold_single_threshold, optimal_point_single_threshold = Youden_J(fpr, tpr, thresholds)

    # Compute optimal point and threshold --- Equal Odds
    gammas_equal_odds, _ = solve_gamma_from_roc_points_equal_odds(fpr_groups, tpr_groups)
    optimal_point_equal_odds = { g: (float(gammas_equal_odds[0]), float(gammas_equal_odds[1])) for g in groups}
    
    # Compute optimal point and threshold --- Equal Opportunity
    gammas_equal_opportunity,_ = solve_gammas_from_roc_points_equal_opportunity(fpr_groups, tpr_groups, pi0, pi1)
    optimal_point_equal_opportunity = { g: tuple(gammas_equal_opportunity[i]) for i, g in enumerate(groups)}

    # Compute optimal point and threshold --- Demographic Parity
    gammas_demographic_parity, _ = solve_gammas_from_roc_points_demographic_parity(fpr_groups, tpr_groups, pi0, pi1)
    optimal_point_demographic_parity = { g: tuple(gammas_demographic_parity[i]) for i, g in enumerate(groups)}

    evaluation_results = {
        "Overall": {
            "thresholds": thresholds,
            "fpr": fpr,
            "tpr": tpr,
        },
        "By_Group": {
            g : {
                "group": g,
                "thresholds": thresholds_by_group,
                "fpr": roc_points_by_group[g]["fpr"],
                "tpr": roc_points_by_group[g]["tpr"],
            } for g in roc_points_by_group
        },
        "Max_Profit": {
            "optimal_point": optimal_point_max_profit,
            "optimal_threshold": optimal_thresholds_max_profit
        },
        "Single_Threshold": {
            "optimal_point": optimal_point_single_threshold,
            "optimal_threshold": optimal_threshold_single_threshold
        },
        "Equal_Odds": {
            "optimal_point": optimal_point_equal_odds
        },
        "Equal_Opportunity": {
            "optimal_point": optimal_point_equal_opportunity
        },
        "Demographic_Parity": {
            "optimal_point": optimal_point_demographic_parity
        }
    }
    return evaluation_results

def plot_results(evaluation_results):
    # Plot single threshold ROC curve
    plot_single_roc_curve(
        fpr=evaluation_results["Overall"]["fpr"],
        tpr=evaluation_results["Overall"]["tpr"],
        optimal_point=evaluation_results["Single_Threshold"]["optimal_point"],
        label = "Roc Curve single threshold"
    )
    
    # Plot max profit ROC curve
    plot_grouped_roc_curves(
        roc_points_by_group=evaluation_results["By_Group"],
        labels_group={0:"Male", 1:"Female"},
        optimal_points_by_group = {g: [evaluation_results["Max_Profit"][g]["optimal point"]] for g in evaluation_results["Max_Profit"]},
        label = "Roc Curve max profit"
    )
    # Plot equal odds ROC curve
    plot_grouped_roc_curves(
        roc_points_by_group=evaluation_results["Max_Profit"],
        labels_group={0:"Male", 1:"Female"},
        optimal_points_by_group= evaluation_results["Equal_Odds"]["optimal_gamma"],
        label = "Roc Curve equal odds"
    )

    # Plot equal opportunity ROC curve
    plot_grouped_roc_curves(
        roc_points_by_group=evaluation_results["Max_Profit"],
        labels_group={0:"Male", 1:"Female"},
        optimal_points_by_group= evaluation_results["Equal_Opportunity"]["optimal_gamma"],
        label = "Roc Curve equal opportunity"
    )
    
    # Plot demographic parity ROC curve
    plot_grouped_roc_curves(
        roc_points_by_group=evaluation_results["Max_profit"],
        labels_group={0:"Male", 1:"Female"},
        optimal_points_by_group= evaluation_results["Demographic_Parity"]["optimal gamma"],
        label = "Roc Curve demographic parity"
    )

def run_fairness_analysis():
    # TODO implement fairness analysis
    pass


if __name__ == "__main__":
    splits, df = load_and_preprocess_data()
    (X_train, y_train), (X_val, y_val, gender_val), test = splits
    model = train_model(X_train, y_train)
    evaluation_results = evaluate_model(model, X_val, y_val, gender_val)
    plot_results(evaluation_results)