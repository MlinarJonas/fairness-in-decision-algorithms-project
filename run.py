import numpy as np 
import matplotlib.pyplot as plt
from cleaning import clean_data
from utils import load_csv_data, make_submission_LR
from preprocessing import preprocess_data
from models import predict_three_layer_nn, ensemble_predictions
from training import train_reg_logistic_regression_weighted, train_weighted_three_layer_nn
from CV_models import cross_validate_model, weighted_lr_model, reg_weighted_lr_model, nn_weighted_model, ensemble_model


def main():
    # Load data
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data")

    # Get column names
    column_names = np.genfromtxt("data/x_train.csv", delimiter=",", max_rows=1, dtype=str)[1:].tolist()

    # Clean data
    x_train_clean, x_test_clean, y_train_clean, column_names = clean_data(
        x_train, x_test, y_train, column_names,
        nan_feature_thresh=0.3, corr_thresh=0.05, high_corr_thresh=0.9, nan_row_thresh=0.05
    )

    # Preprocess cleaned data
    x_train_final, x_test_final, y_train_bin = preprocess_data(
        x_train_clean, x_test_clean, y_train_clean, column_names
    )

    # Train best model (weighted L2-regularized logistic regression)
    
    w, loss = train_reg_logistic_regression_weighted(
        x_train_final, y_train_bin, lambda_=1e-6, gamma=0.5, max_iters=2000
    )
    print("Training finished")
    
    # Generate submission
    y_test_competition = make_submission_LR(
        w, x_test_final, test_ids, filename="submission_final.csv"
    )
    print("Submission saved to submission_final.csv")
    
    """
    # Prediction with neural network 
    
    params_3 = train_weighted_three_layer_nn(
    x_train_final, y_train_bin,
    n_hidden1=128, n_hidden2=64,
    lr=0.01, lambda_=1e-4, epochs=500, class_weight=True)

    y_pred_3, y_prob_3 = predict_three_layer_nn(x_test_final, params_3)
    
    # Map to -1/1 for submission
    y_test_competition = np.where(y_pred_3 == 0, -1, 1)

    # Save submission
    submission_array = np.column_stack((test_ids, y_test_competition))
    np.savetxt("submission_weighted_3_layers_nn_new.csv", submission_array, fmt='%d', delimiter=",", 
           header="Id,Prediction", comments='')

    print("Submission file saved as 'submission_weighted_3_layers_nn_new.csv'")
    
    # Prediction with ensembling
    
    y_pred_ens, y_prob_ens, y_sub_ens = ensemble_predictions(
    x_test_final, params_3, w,
    weight_nn=0.7, weight_logreg=0.3)

    # Save submission
    submission_array = np.column_stack((test_ids, y_sub_ens))
    np.savetxt("submission_ensemble_nn_logreg_new.csv", submission_array, fmt='%d', delimiter=",",
           header="Id,Prediction", comments='')

    

    ### Compare all models with CV
    models = {
        "Weighted LR": weighted_lr_model,
        "Reg. Weighted LR": reg_weighted_lr_model,
        "NN": nn_weighted_model
        "Ensembling": ensemble_model
    }

    results = {}
    for name, func in models.items():
        print(f"Running CV for {name}...")
        metrics = cross_validate_model(y_train_bin, x_train_final, func, k_fold=5)
        results[name] = metrics
        
    # Plot results
    metrics_to_plot = ["accuracy", "f1"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(12, 5))

    for i, metric in enumerate(metrics_to_plot):
        data = [results[m][metric] for m in models]
        axes[i].boxplot(data, labels=models.keys(), patch_artist=True)
        axes[i].set_title(f"{metric.upper()} Comparison")
        axes[i].set_ylabel(metric.capitalize())
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        # Add mean annotation
        for j, scores in enumerate(data):
            mean = np.mean(scores)
            axes[i].text(j + 1, mean + 0.002, f"{mean:.3f}",
                        ha='center', color='red', fontsize=9)

    plt.tight_layout()
    plt.show()
    
    # Print summary table of results
    print("Model Performance Summary:")
    print(f"{'Model':<25} {'Accuracy (mean±std)':<25} {'F1 (mean±std)':<25}")
    for name, res in results.items():
        acc_mean, acc_std = np.mean(res["accuracy"]), np.std(res["accuracy"])
        f1_mean, f1_std = np.mean(res["f1"]), np.std(res["f1"])
        print(f"{name:<25} {acc_mean:.3f}±{acc_std:.3f} {f1_mean:>15.3f}±{f1_std:.3f}")

    """

if __name__ == "__main__":
    main()



