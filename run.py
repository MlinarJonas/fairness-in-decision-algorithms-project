import numpy as np 
import matplotlib.pyplot as plt
# from cleaning import clean_data
# from preprocessing import preprocess_data
from logistic_regression import reg_logistic_regression_weighted
from training import train_reg_logistic_regression_weighted
from CV_models import cross_validate_model, reg_weighted_lr_model
from optimize_gamma import solve_gamma_from_roc_points, get_roc_points


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
    
    #w, loss = train_reg_logistic_regression_weighted(
    #    x_train_final, y_train_bin, lambda_=1e-6, gamma=0.5, max_iters=2000
    #)
    #print("Training finished")
    
    # optimize gamma
    fpr_groups, tpr_groups=get_roc_points()
    gamma, lambdas = solve_gamma_from_roc_points(fpr_groups, tpr_groups)
    print("Gamma", gamma, "lambdas", lambdas)
    # Generate submission
    #y_test_competition = make_submission_LR(
    #    w, x_test_final, test_ids, filename="submission_final.csv"
    #)
    #print("Submission saved to submission_final.csv")

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
    

if __name__ == "__main__":
    main()



