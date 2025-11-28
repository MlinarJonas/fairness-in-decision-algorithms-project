import numpy as np 
import matplotlib.pyplot as plt
# from cleaning import clean_data
# from preprocessing import preprocess_data
from logistic_regression import reg_logistic_regression_weighted
from training import train_reg_logistic_regression_weighted
from CV_models import cross_validate_model, reg_weighted_lr_model


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
    

if __name__ == "__main__":
    main()



