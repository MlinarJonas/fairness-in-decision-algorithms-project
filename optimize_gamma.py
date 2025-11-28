import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from data_processing import get_data
from data_split import train_val_test_split
from logistic_regression import LogisticRegression
from utils import scores

def get_roc_points_and_class_distribution(threshold_limit=1000):
    # Load and preprocess data
    df = get_data()

    # Split data into train, validation, and test sets
    train, val, test = train_val_test_split(df)

    # Prepare training data
    X_train = train.drop(columns=['pass_bar']).to_numpy()
    y_train = train['pass_bar'].to_numpy()

    X_val = val.drop(columns=['pass_bar']).to_numpy()
    y_val = val['pass_bar'].to_numpy()

    # Keep gender column (protected attribute A)
    gender_val = val['gender'].to_numpy()

    # thresholds for ROC curve
    thresholds = np.linspace(0, 1, threshold_limit)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Prepare lists for ROC for men and women
    tprs_men, fprs_men = [], []
    tprs_women, fprs_women = [], []

    # --- Compute ROC curves ----------------------------------------------------
    for threshold in thresholds:

        # Predict on validation set using threshold
        y_pred_val = model.predict(X_val, threshold=threshold)

        # ---- MEN ----
        mask_men = (gender_val == 0)
        y_true_men = y_val[mask_men]
        y_pred_men = y_pred_val[mask_men]

        _, _, recall_men, fpr_men, _ = scores(y_pred_men, y_true_men)

        tprs_men.append(recall_men)
        fprs_men.append(fpr_men)

        # ---- WOMEN ----
        mask_women = (gender_val == 1)
        y_true_women = y_val[mask_women]
        y_pred_women = y_pred_val[mask_women]

        _, _, recall_women, fpr_women, _ = scores(y_pred_women, y_true_women)

        tprs_women.append(recall_women)
        fprs_women.append(fpr_women)

    # Convert to numpy arrays
    tprs_men = np.array(tprs_men)
    fprs_men = np.array(fprs_men)
    tprs_women = np.array(tprs_women)
    fprs_women = np.array(fprs_women)

    fpr_groups = [fprs_men, fprs_women]
    tpr_groups = [tprs_men, tprs_women]

    # --- Compute class distribution (pi0, pi1) ----------------------------------
    # pi0[a] = P(A=a, Y=0)
    # pi1[a] = P(A=a, Y=1)
    
    N = len(y_val)
    if N == 0:
        raise ValueError("Validation set is empty; cannot compute class distribution.")

    pi0 = []
    pi1 = []

    for a in [0, 1]:  # men, women
        mask_a = (gender_val == a)
        n_a0 = np.sum(mask_a & (y_val == 0))  # count of (A=a, Y=0)
        n_a1 = np.sum(mask_a & (y_val == 1))  # count of (A=a, Y=1)

        pi0.append(n_a0 / N)
        pi1.append(n_a1 / N)

    pi0 = np.array(pi0, dtype=float)
    pi1 = np.array(pi1, dtype=float)

    return fpr_groups, tpr_groups, pi0, pi1


def solve_gamma_from_roc_points_equal_odds(fpr_groups, tpr_groups, l10=1, l01=1):
    """
    Solve:
        min_{γ in ∩_a D_a} γ0 * l(1,0) + (1 - γ1) * l(0,1)
    where for each group a,
        D_a = conv{ (FPR_a,i, TPR_a,i) : ROC points for group a }.

    Parameters
    ----------
    fpr_groups : list of 1D arrays
        fpr_groups[a] is an array of FPR values for group a (length m_a).
    tpr_groups : list of 1D arrays
        tpr_groups[a] is an array of TPR values for group a (length m_a).
        Must match lengths of fpr_groups[a].
    l10 : float
        Loss l(1,0): loss for predicting 1 when true label is 0 (false positive).
    l01 : float
        Loss l(0,1): loss for predicting 0 when true label is 1 (false negative).

    Returns
    -------
    gamma : np.ndarray, shape (2,)
        The optimal γ = (γ0, γ1).
    lambdas : list of 1D arrays
        lambdas[a] are the convex coefficients (same length as fpr_groups[a])
        representing γ as a convex combination of ROC points in group a.
    """
    fpr_groups = [np.asarray(f) for f in fpr_groups]
    tpr_groups = [np.asarray(t) for t in tpr_groups]

    num_groups = len(fpr_groups)
    m_list = [len(f) for f in fpr_groups]  # m_a for each group a

    # Total number of lambda variables = sum_a m_a
    total_lambdas = sum(m_list)

    # Decision variables: [γ0, γ1, λ_1^(0)...λ_m0^(0), λ_1^(1)..., ...]
    n_vars = 2 + total_lambdas

    # Objective: minimize γ0 * l10 + (1 - γ1) * l01
    # Equivalent to: minimize γ0 * l10 - γ1 * l01 (constant l01 is dropped)
    c = np.zeros(n_vars)
    c[0] = l10      # coefficient for γ0
    c[1] = -l01     # coefficient for γ1

    # Build equality constraints:
    # For each group a, we enforce:
    #   γ0 = sum_i λ_i^(a) * FPR_a,i
    #   γ1 = sum_i λ_i^(a) * TPR_a,i
    #   sum_i λ_i^(a) = 1
    A_eq_rows = []
    b_eq = []

    # Offset index where each group's λ-block starts
    lambda_start = 2
    for a in range(num_groups):
        fpr = fpr_groups[a]
        tpr = tpr_groups[a]
        m_a = m_list[a]

        # indices of λ^(a) in the full x vector
        offset = lambda_start
        lam_indices = np.arange(offset, offset + m_a)

        # 1) γ0 - sum_i λ_i^(a) * FPR_a,i = 0
        row = np.zeros(n_vars)
        row[0] = 1.0
        row[lam_indices] = -fpr
        A_eq_rows.append(row)
        b_eq.append(0.0)

        # 2) γ1 - sum_i λ_i^(a) * TPR_a,i = 0
        row = np.zeros(n_vars)
        row[1] = 1.0
        row[lam_indices] = -tpr
        A_eq_rows.append(row)
        b_eq.append(0.0)

        # 3) sum_i λ_i^(a) = 1
        row = np.zeros(n_vars)
        row[lam_indices] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)

        lambda_start += m_a

    A_eq = np.vstack(A_eq_rows)
    b_eq = np.array(b_eq)

    # Bounds:
    #   0 <= γ0 <= 1
    #   0 <= γ1 <= 1
    #   λ_i^(a) >= 0 (no explicit upper bound)
    bounds = []
    bounds.append((0.0, 1.0))  # γ0
    bounds.append((0.0, 1.0))  # γ1
    for _ in range(total_lambdas):
        bounds.append((0.0, None))  # λ >= 0

    # Solve LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")

    x = res.x
    gamma = x[0:2]

    # Extract λ blocks per group
    lambdas = []
    lambda_start = 2
    for m_a in m_list:
        lam = x[lambda_start:lambda_start + m_a]
        # Normalize to sum 1 (just to clean numerical noise)
        lam = np.maximum(lam, 0)
        s = lam.sum()
        if s > 0:
            lam /= s
        lambdas.append(lam)
        lambda_start += m_a

    return gamma, lambdas


def solve_gammas_from_roc_points_equal_opportunity(
    fpr_groups,
    tpr_groups,
    pi0,
    pi1,
    l10=1.0,
    l01=1.0,
):
    """
    Equal Opportunity:
        Find per-group operating points γ_a = (FPR_a, TPR_a) such that
        all groups share the same TPR, and the expected loss is minimized.

    We solve:
        min  sum_a [ pi0[a]*FPR_a*l10 + pi1[a]*(1-TPR_a)*l01 ]
      s.t.
        For each group a:
          (FPR_a, TPR_a) is a convex combination of that group's ROC points
        TPR_0 = TPR_a for all a (Equal Opportunity)
        λ_{a,i} >= 0, sum_i λ_{a,i} = 1

    Parameters
    ----------
    fpr_groups : list of 1D arrays
        fpr_groups[a][i] = FPR of ROC point i for group a.
    tpr_groups : list of 1D arrays
        tpr_groups[a][i] = TPR of ROC point i for group a.
    pi0 : 1D array-like of length num_groups
        pi0[a] ∝ P(A=a, Y=0) or any nonnegative weights for negatives per group.
    pi1 : 1D array-like of length num_groups
        pi1[a] ∝ P(A=a, Y=1) or any nonnegative weights for positives per group.
    l10 : float
        Loss l(1,0): false positive cost.
    l01 : float
        Loss l(0,1): false negative cost.

    Returns
    -------
    gammas : np.ndarray, shape (num_groups, 2)
        gammas[a] = (FPR_a, TPR_a) for each group a.
    lambdas : list of 1D arrays
        lambdas[a] are convex coefficients for group a such that
            γ_a = sum_i lambdas[a][i] * (FPR_{a,i}, TPR_{a,i})
    """
    fpr_groups = [np.asarray(f) for f in fpr_groups]
    tpr_groups = [np.asarray(t) for t in tpr_groups]

    num_groups = len(fpr_groups)
    if num_groups < 2:
        raise ValueError("Need at least two groups for Equal Opportunity.")

    pi0 = np.asarray(pi0, dtype=float)
    pi1 = np.asarray(pi1, dtype=float)
    if pi0.shape[0] != num_groups or pi1.shape[0] != num_groups:
        raise ValueError("pi0 and pi1 must have length equal to number of groups.")

    m_list = [len(f) for f in fpr_groups]
    total_lambdas = sum(m_list)

    # Decision variables: all λ's concatenated
    n_vars = total_lambdas
    c = np.zeros(n_vars)

    # Build linear objective in λ:
    # Risk = sum_a sum_i λ_{a,i} [ pi0[a]*fpr_{a,i}*l10 + pi1[a]*(1 - tpr_{a,i})*l01 ]
    #      = constant + sum_a sum_i λ_{a,i} [ pi0[a]*fpr_{a,i}*l10 - pi1[a]*tpr_{a,i}*l01 ]
    offset = 0
    for a in range(num_groups):
        fpr = fpr_groups[a]
        tpr = tpr_groups[a]
        m_a = m_list[a]
        print(m_a)

        c[offset:offset + m_a] = pi0[a] * fpr * l10 - pi1[a] * tpr * l01
        offset += m_a

    A_eq_rows = []
    b_eq = []

    # 1) Equal TPR across groups: use group 0 as reference
    #    For each a>0:
    #      sum_i λ_{0,i} tpr_{0,i} - sum_j λ_{a,j} tpr_{a,j} = 0
    offset0_start = 0
    offset = 0
    # precompute cumulative offsets to access blocks cleanly
    offsets = np.cumsum([0] + m_list)

    for a in range(1, num_groups):
        row = np.zeros(n_vars)
        # group 0 block
        row[offsets[0]:offsets[1]] = tpr_groups[0]
        # group a block
        row[offsets[a]:offsets[a+1]] = -tpr_groups[a]
        A_eq_rows.append(row)
        b_eq.append(0.0)

    # 2) Convex hull for each group: sum_i λ_{a,i} = 1
    for a in range(num_groups):
        row = np.zeros(n_vars)
        row[offsets[a]:offsets[a+1]] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)

    A_eq = np.vstack(A_eq_rows)
    b_eq = np.array(b_eq)

    bounds = [(0.0, None)] * n_vars  # λ >= 0

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Equal Opportunity LP failed: {res.message}")

    lam_all = res.x

    # Split λ and compute per-group gammas
    lambdas = []
    gammas = []
    for a in range(num_groups):
        lam = lam_all[offsets[a]:offsets[a+1]]
        lam = np.maximum(lam, 0)
        s = lam.sum()
        if s > 0:
            lam /= s
        lambdas.append(lam)

        fpr_a = np.dot(lam, fpr_groups[a])
        tpr_a = np.dot(lam, tpr_groups[a])
        gammas.append(np.array([fpr_a, tpr_a]))

    gammas = np.vstack(gammas)
    return gammas, lambdas

fpr_groups, tpr_groups, pi0, pi1= get_roc_points_and_class_distribution()
print(solve_gammas_from_roc_points_equal_opportunity(fpr_groups, tpr_groups, pi0, pi1))