import numpy as np

def find_threshold_on_single_roc(fprs, tprs, thresholds, gamma, tol=1e-8):
    """
    Given a single ROC curve (fprs, tprs, thresholds) and a target point gamma,
    return a threshold representation:
       - deterministic: T
       - randomized: { 'T_a': T_a, 'T_b': T_b, 'p': alpha }
    The pair of points used for the randomized version is chosen as the one whose
    segment contains gamma AND whose midpoint is closest (Euclidean distance) to gamma.
    """
    fprs = np.asarray(fprs, float)
    tprs = np.asarray(tprs, float)
    thresholds = np.asarray(thresholds, float)
    gamma = np.asarray(gamma, float)

    # --- STEP 1: Try deterministic match (i.e., gamma exactly equal to some ROC point) ---
    for i, (x, y) in enumerate(zip(fprs, tprs)):
        if np.allclose([x, y], gamma, atol=tol, rtol=0):
            return float(thresholds[i])  # deterministic threshold

    # --- STEP 2: Search pairs whose segment contains gamma ---
    candidates = []   # will store (distance_to_gamma, i, j, alpha)

    for i in range(len(fprs)):
        for j in range(i+1, len(fprs)):
            a = np.array([fprs[i], tprs[i]], float)
            b = np.array([fprs[j], tprs[j]], float)

            # Solve for alpha in gamma = (1-alpha)*a + alpha*b
            ab = b - a
            denom = ab[0] if abs(ab[0]) > tol else ab[1]
            if abs(denom) < tol:
                continue

            if abs(ab[0]) > tol:
                alpha = (gamma[0] - a[0]) / ab[0]
            else:
                alpha = (gamma[1] - a[1]) / ab[1]

            # Check alpha in [0,1] and that the reconstructed point equals gamma
            if 0 - tol <= alpha <= 1 + tol:
                recon = (1 - alpha) * a + alpha * b
                if np.linalg.norm(recon - gamma) < 1e-6:
                    midpoint = 0.5 * (a + b)
                    dist = np.linalg.norm(midpoint - gamma)
                    candidates.append((dist, i, j, float(alpha)))

    # --- STEP 3: Choose the closest-valid pair ---
    if len(candidates) == 0:
        raise ValueError("Gamma does not lie on any segment of this ROC curve.")

    candidates.sort(key=lambda x: x[0])
    _, i, j, alpha = candidates[0]

    # --- STEP 4: Build randomized threshold ---
    # If alpha is 0 or 1 exactly -> deterministic, but this rarely happens.
    if alpha <= tol:
        return float(thresholds[i])
    if alpha >= 1 - tol:
        return float(thresholds[j])

    return {
        'T_a': float(thresholds[i]),
        'T_b': float(thresholds[j]),
        'p': float(alpha)
    }
