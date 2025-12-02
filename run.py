import numpy as np 
import matplotlib.pyplot as plt
from metrics.threshold import find_threshold_on_single_roc
from data.data_processing import download_dataset, load_dataset, process_dataset, prepare_data_for_training
from training.train import train_logistic_regression
from metrics.roc import compute_roc_points_by_group, compute_roc_points
from utils.utils import Youden_J
from utils.plots import plot_single_roc_curve, plot_grouped_roc_curves

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
    y_score = model.predict_proba(X_val)
    thresholds, fpr, tpr = compute_roc_points(y_val, y_score)
    thresholds_by_group, roc_points_by_group = compute_roc_points_by_group(y_val, y_score, gender_val)
    optimal_threshold, optimal_point = Youden_J(fpr, tpr, thresholds)
    evaluation_results = {
        "overall": {
            "thresholds": thresholds,
            "fpr": fpr,
            "tpr": tpr,
            "optimal threshold": optimal_threshold,
            "optimal point": optimal_point
        },
        "by_group": {
            g : {
                "group": g,
                "thresholds": thresholds_by_group,
                "fpr": roc_points_by_group[g]["fpr"],
                "tpr": roc_points_by_group[g]["tpr"],
                # "optimal point": TODO start here for equal odds
            } for g in roc_points_by_group
        }
    } 
    return evaluation_results

def plot_results(evaluation_results):
    # Plot overall ROC curve
    plot_single_roc_curve(
        fpr=evaluation_results["overall"]["fpr"],
        tpr=evaluation_results["overall"]["tpr"],
        optimal_point=evaluation_results["overall"]["optimal point"]
    )
    
    plot_grouped_roc_curves(
        roc_points_by_group=evaluation_results["by_group"],
        labels_group={0:"Male", 1:"Female"}
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


def main():
    # Load data
    # x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data")

    # Get column names
    # column_names = np.genfromtxt("data/x_train.csv", delimiter=",", max_rows=1, dtype=str)[1:].tolist()

    # Clean data
    #x_train_clean, x_test_clean, y_train_clean, column_names = clean_data(
    #    x_train, x_test, y_train, column_names,
    #    nan_feature_thresh=0.3, corr_thresh=0.05, high_corr_thresh=0.9, nan_row_thresh=0.05
    #)

    # Preprocess cleaned data
    #x_train_final, x_test_final, y_train_bin = preprocess_data(
    #    x_train_clean, x_test_clean, y_train_clean, column_names
    #)

    # Train best model (weighted L2-regularized logistic regression)
    
    w, loss = train_reg_logistic_regression_weighted(
        x_train_final, y_train_bin, lambda_=1e-6, gamma=0.5, max_iters=2000
    )
    print("Training finished")
    
    # Generate submission
    #y_test_competition = make_submission_LR(
    #    w, x_test_final, test_ids, filename="submission_final.csv"
    #)
    #print("Submission saved to submission_final.csv")

    thresholds = np.linspace(0, 1, 1000)
    T_0 = find_threshold_on_single_roc(fpr_groups[0], tpr_groups[0], thresholds, gamma, tol=1e-12)
    print("T_0:", T_0)
    T_1 = find_threshold_on_single_roc(fpr_groups[1], tpr_groups[1], thresholds, gamma, tol=1e-12)
    print("T_1:", T_1)

    # Convert to numpy arrays
    tprs_men = tpr_groups[0]
    fprs_men = fpr_groups[0]
    tprs_women = tpr_groups[1]
    fprs_women = fpr_groups[1]
    
    
    # Plot ROC curves
    plt.figure()
    plt.plot(fprs_men, tprs_men, label='Men ROC')
    plt.plot(fprs_women, tprs_women, label='Women ROC')
    plt.scatter(gamma[0], gamma[1], marker="x", color="red", label="Gamma")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve by Gender')
    plt.legend()
    plt.show()



    # Extract ROC points for plotting
    (T0, alpha0, gamma0, idx0) = T_0
    (T1, alpha1, gamma1, idx1) = T_1

    roc0_points = [(fprs_men[i], tprs_men[i]) for i in idx0]
    roc1_points = [(fprs_women[i], tprs_women[i]) for i in idx1]

    plt.figure()

    # ROC curves
    plt.plot(fprs_men, tprs_men, label='Men ROC')
    plt.plot(fprs_women, tprs_women, label='Women ROC')

    # Gamma
    plt.scatter(gamma[0], gamma[1], c="red", s=120, marker="x", label="Gamma")

    # ----- Group 0 convex hull points -----
    for (x, y) in roc0_points:
        plt.scatter(x, y, color="blue")
    # connect hull (segment or triangle)
    xs0 = [p[0] for p in roc0_points] + [roc0_points[0][0]]
    ys0 = [p[1] for p in roc0_points] + [roc0_points[0][1]]
    plt.plot(xs0, ys0, color="blue", linestyle="--", label="Group 0 combination")

    # ----- Group 1 convex hull points -----
    for (x, y) in roc1_points:
        plt.scatter(x, y, color="orange")
    xs1 = [p[0] for p in roc1_points] + [roc1_points[0][0]]
    ys1 = [p[1] for p in roc1_points] + [roc1_points[0][1]]
    plt.plot(xs1, ys1, color="orange", linestyle="--", label="Group 1 combination")

    # Diagonal
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve by Gender with Gamma and Convex Combination Points")
    plt.legend()
    plt.show()




