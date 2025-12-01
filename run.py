import numpy as np 
import matplotlib.pyplot as plt
# from cleaning import clean_data
# from preprocessing import preprocess_data
from logistic_regression import reg_logistic_regression_weighted
from train import train_reg_logistic_regression_weighted
from CV_models import cross_validate_model, reg_weighted_lr_model
from optimize_gamma import solve_gamma_from_roc_points, get_roc_points
from threshold import find_threshold_on_single_roc


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



    

if __name__ == "__main__":
    main()



