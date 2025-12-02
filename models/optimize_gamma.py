import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from data.data_processing import get_data
from data.data_split import train_val_test_split
from models.logistic_regression import LogisticRegression
from utils.utils import scores

def class_distribution_by_group(y_val, group_val):
    
    N = len(y_val)
    groups = np.unique(group_val) # unique group values

    distribution_by_group = {}

    for g in groups:
        mask = (group_val == g)
        n_g = mask.sum()  # size of group g

        # Joint probabilities P(A=g, Y=y)
        p_joint_y0 = np.sum(mask & (y_val == 0)) / N
        p_joint_y1 = np.sum(mask & (y_val == 1)) / N

        # Conditional probabilities  P(Y=y | A=g)
        if n_g > 0:
            p_cond_y0 = np.sum(mask & (y_val == 0)) / n_g
            p_cond_y1 = np.sum(mask & (y_val == 1)) / n_g

        else:
            p_cond_y0 = p_cond_y1 = np.nan

            #n_a0 = np.sum(mask_a & (y_val == 0))  # count of (A=a, Y=0)
            #n_a1 = np.sum(mask_a & (y_val == 1))  # count of (A=a, Y=1)

            #pi0 = np.array(pi0, dtype=float)/np.sum(pi0)
            #pi1 = np.array(pi1, dtype=float)/np.sum(pi1)

        distribution_by_group[g] = {
            "joint": {"P(A=g,Y=0)": p_joint_y0, "P(A=g,Y=1)": p_joint_y1},
            "conditional": {"P(Y=0|A=g)": p_cond_y0, "P(Y=1|A=g)": p_cond_y1},
            "count": int(n_g)
        }

    return distribution_by_group


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

def find_optimal_gamma(roc_points, l01 = 1, l10 = 1):
    """
    roc_points: list of tuples (FPR, TPR) describing the ROC curve.
                They do NOT have to be sorted.
                
    loss: function taking a pair (y_true, y_pred) in {(1,0),(0,1)} and returning l(y_true, y_pred)
          Example:
              def loss(pair):
                  if pair == (1,0): return c_fp
                  if pair == (0,1): return c_fn
                  
    Returns:
        gamma_star: optimal point (gamma_0, gamma_1)
        value: the minimal loss value
        index: index of the optimal ROC point in the sorted list
    """

    # Sort ROC points by FPR (conventional ROC ordering)
    roc = sorted(roc_points, key=lambda p: p[0])

    best_value = float('inf')
    best_gamma = None
    best_idx = None

    for i, (fpr, tpr) in enumerate(roc):
        fnr = 1 - tpr
        value = fpr * l10 + fnr * l01

        if value < best_value:
            best_value = value
            best_gamma = (fpr, fnr)
            best_idx = i

    return best_gamma, best_value, best_idx

def solve_gammas_from_roc_points_demographic_parity(
    fpr_groups,
    tpr_groups,
    pi0,
    pi1,
    l10=1.0,
    l01=1.0,
):
    """
    Demographic Parity:
        Selection rate SR_a must be equal across groups.

    We find per-group operating points gamma_a = (FPR_a, TPR_a)
    such that:
        SR_0 = SR_1 = ... = SR_K
    and the total expected loss is minimized.

    Returns: gammas (group-specific FPR/TPR) and λ vectors.
    """
    fpr_groups = [np.asarray(f) for f in fpr_groups]
    tpr_groups = [np.asarray(t) for t in tpr_groups]

    num_groups = len(fpr_groups)

    pi0 = np.asarray(pi0, float)
    pi1 = np.asarray(pi1, float)

    m_list = [len(f) for f in fpr_groups]
    total_lambdas = sum(m_list)

    # Decision variables = all λ's concatenated
    n_vars = total_lambdas
    c = np.zeros(n_vars)

    # Linear risk: R = sum_a sum_i λ_{a,i} (pi0*FPR*l10 - pi1*TPR*l01)
    offset = 0
    for a in range(num_groups):
        fpr = fpr_groups[a]
        tpr = tpr_groups[a]
        m_a = m_list[a]

        c[offset:offset+m_a] = pi0[a]*fpr*l10 - pi1[a]*tpr*l01
        offset += m_a

    # Equality constraints: DP + convexity
    A_eq_rows = []
    b_eq = []

    # Precompute block offsets
    offsets = np.cumsum([0] + m_list)

    # Define selection-rate term per point
    s = []
    for a in range(num_groups):
        s.append(pi0[a] * fpr_groups[a] + pi1[a] * tpr_groups[a])

    # DP constraint: SR_0 = SR_a for all a
    for a in range(1, num_groups):
        row = np.zeros(n_vars)
        row[offsets[0]:offsets[1]] = s[0]
        row[offsets[a]:offsets[a+1]] = -s[a]
        A_eq_rows.append(row)
        b_eq.append(0.0)

    # Convex weights sum to 1 for each group
    for a in range(num_groups):
        row = np.zeros(n_vars)
        row[offsets[a]:offsets[a+1]] = 1.0
        A_eq_rows.append(row)
        b_eq.append(1.0)

    A_eq = np.vstack(A_eq_rows)
    b_eq = np.array(b_eq)

    bounds = [(0.0, None)] * n_vars

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        raise RuntimeError(f"Demographic Parity LP failed: {res.message}")

    # Extract λ and produce group gammas
    lam_all = res.x
    lambdas = []
    gammas = []

    for a in range(num_groups):
        lam = lam_all[offsets[a]:offsets[a+1]]
        lam = np.maximum(lam, 0)
        lam /= lam.sum()

        FPR_a = np.dot(lam, fpr_groups[a])
        TPR_a = np.dot(lam, tpr_groups[a])

        lambdas.append(lam)
        gammas.append([FPR_a, TPR_a])

    return np.array(gammas), lambdas