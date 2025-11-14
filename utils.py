import numpy as np
import csv
import os

### ------ from helpers.py -------- ###
def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids
def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

### ----- our own utils.py functions ----- ###
def compute_feature_correlations(X, y):
    """
    Compute absolute correlation between each feature in X and target y.
    Handles NaNs and returns an array of shape (n_features,).
    """
    corrs = np.zeros(X.shape[1])
    y_mean, y_std = np.mean(y), np.std(y)

    for j in range(X.shape[1]):
        xj = X[:, j]
        mask = ~np.isnan(xj)
        if np.sum(mask) < 2:
            corrs[j] = 0
            continue

        xj = xj[mask]
        yj = y[mask]
        x_mean, x_std = np.mean(xj), np.std(xj)
        if x_std < 1e-12 or y_std < 1e-12:
            corrs[j] = 0
            continue

        cov = np.mean((xj - x_mean) * (yj - y_mean))
        corrs[j] = abs(cov / (x_std * y_std))
    return corrs
def remove_highly_correlated_features(x, threshold=0.9):
    """
    Remove highly correlated features.
    Returns reduced X and indices of kept columns.
    """
    corr_matrix = np.corrcoef(x, rowvar=False)
    n_features = corr_matrix.shape[0]
    to_drop = set()

    for i in range(n_features):
        if i in to_drop:
            continue
        for j in range(i + 1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)

    kept_indices = [i for i in range(n_features) if i not in to_drop]
    x_reduced = x[:, kept_indices]
    return x_reduced, kept_indices
def one_hot_encode_column(train_col, test_col):
        train_col = train_col.astype(int)
        test_col = test_col.astype(int)
        categories = np.unique(train_col)
        n_cat = len(categories)
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        train_idx = np.vectorize(cat_to_idx.get)(train_col)
        train_encoded = np.zeros((len(train_col), n_cat))
        train_encoded[np.arange(len(train_col)), train_idx] = 1

        test_idx = np.array([cat_to_idx.get(v, n_cat) for v in test_col])
        n_cat_test = n_cat + 1 if np.any(test_idx == n_cat) else n_cat
        test_encoded = np.zeros((len(test_col), n_cat_test))
        test_encoded[np.arange(len(test_col)), test_idx] = 1
        if n_cat_test > n_cat:
            train_encoded = np.pad(train_encoded, ((0, 0), (0, 1)), constant_values=0)

        return train_encoded, test_encoded
def classify_features(x, column_names, manual_num=None, cat_unique_threshold=20):
    """
    Classify features into numeric, categorical and mixed.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix (rows=samples, columns=features)
    column_names : list of str
        Names of columns
    manual_num : list of str, optional
        Column names that should always be numeric (even if classified as mixed)
    cat_unique_threshold : int
        Max unique values to consider a feature categorical

    Returns
    -------
    num_features : list of int
        Indices of numeric features
    cat_features : list of int
        Indices of categorical features
    mixed_features : list of int
        Indices of mixed/coded features
    """
    num_features, cat_features, mixed_features = [], [], []

    for i in range(x.shape[1]):
        col = x[:, i]
        col_nonan = col[~np.isnan(col)]
        unique_vals = np.unique(col_nonan)

        if len(unique_vals) <= cat_unique_threshold:
            cat_features.append(i)
        elif np.any((unique_vals >= 70) & (unique_vals <= 9999)):
            mixed_features.append(i)
        else:
            num_features.append(i)

    # Manual correction for numeric columns misclassified as mixed
    if manual_num is not None:
        manual_num_idx = [column_names.index(name) for name in manual_num if name in column_names]
        for i in manual_num_idx:
            if i in mixed_features:
                mixed_features.remove(i)
            if i not in num_features:
                num_features.append(i)

    return num_features, cat_features, mixed_features
def recode_mixed_features(x_train, x_test, mixed_features, column_names):
    """
    Recode special survey codes for mixed features.

    Parameters
    ----------
    x_train, x_test : np.ndarray
        Feature matrices
    mixed_features : list of int
        Indices of mixed/coded features
    column_names : list of str
        Column names

    Returns
    -------
    x_train_recoded, x_test_recoded : np.ndarray
        Recoded feature matrices
    """
    x_train_recoded = x_train.astype(float)
    x_test_recoded = x_test.astype(float)

    special_codes_nan = [77, 98, 7777, 777, 99, 9999, 999, 99999, 99900, 99000]
    special_codes_zero = [88, 888, 555]
    special_code_half = [300]

    for i in mixed_features:
        col_train = x_train_recoded[:, i]
        col_test = x_test_recoded[:, i]

        # special exceptions for SCNTWRK1/SCNTLWK1
        if column_names[i] in ['SCNTWRK1', 'SCNTLWK1']:
            col_train[col_train == 97] = np.nan
            col_test[col_test == 97] = np.nan
            col_train[col_train == 98] = 0
            col_test[col_test == 98] = 0
            col_train[col_train == 99] = np.nan
            col_test[col_test == 99] = np.nan

        # Apply general recoding
        for code in special_codes_nan:
            col_train[col_train == code] = np.nan
            col_test[col_test == code] = np.nan
        for code in special_codes_zero:
            col_train[col_train == code] = 0
            col_test[col_test == code] = 0
        for code in special_code_half:
            col_train[col_train == code] = 0.5
            col_test[col_test == code] = 0.5

        x_train_recoded[:, i] = col_train
        x_test_recoded[:, i] = col_test

    return x_train_recoded, x_test_recoded
def sigmoid(z):
    """Compute the sigmoid function"""
    return 1 / (1 + np.exp(-z))
def make_submission_LR(w, x_test, test_ids, filename="submission.csv", threshold=0.5):
    """
    Generate predictions from model weights, map to competition format and save submission.

    Args:
        w : np.ndarray
            Weight vector from trained model.
        x_test : np.ndarray
            Test feature matrix (including intercept column).
        test_ids : np.ndarray
            Array of test IDs to include in submission.
        filename : str, optional
            File name for submission CSV (default "submission.csv").
        threshold : float, optional
            Threshold for classifying probabilities into 0/1 (default 0.5).

    Returns:
        y_test_competition : np.ndarray
            Predictions in competition format (-1/1).
    """
    # Compute predicted probabilities
    y_pred_proba = sigmoid(x_test @ w) 

    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Map to -1/1 for competition
    y_test_competition = np.where(y_pred == 0, -1, 1)

    # Save submission
    submission_array = np.column_stack((test_ids, y_test_competition))
    np.savetxt(filename, submission_array, fmt='%d', delimiter=",", 
               header="Id,Prediction", comments='')

    return y_test_competition
def f1_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    if TP + FP == 0 or TP + FN == 0:
        return 0.0  # avoid division by zero
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
def build_k_indices(y, k_fold, seed):
    np.random.seed(seed)
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

