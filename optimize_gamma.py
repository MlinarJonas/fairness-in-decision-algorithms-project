import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from data_processing import get_data
from data_split import train_val_test_split
from logistic_regression import LogisticRegression
from utils import scores
from roc import get_roc_points


def solve_gamma_from_roc_points(fpr_groups, tpr_groups, l10=1, l01=1):
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



roc_points = get_roc_points()

optimzed = solve_gamma_from_roc_points(roc_points[0], roc_points[1], l10=1, l01=1)

print(optimzed)