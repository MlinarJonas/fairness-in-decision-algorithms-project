import itertools
import numpy as np

# ---------------------------
#  Barycentric for 2 points
# ---------------------------

def barycentric_segment(g, p1, p2, tol=1e-12):

    """
    Function that verifies whether the point g lies 
    on the segement between the points p1 and p2 and 
    returns the coefficients alpha and beta such that
    g = alpha*p1 + beta*p2 if it is the case.
    """
    g = np.asarray(g) 
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    v = p2 - p1 # v is the vector from p1 to p2
    denom = np.dot(v, v) # norm of v
    if denom < tol:
        return None  # identical points

    # Solve g = (1 - alpha)*p1 + alpha*p2 
    alpha = np.dot(g - p1, v) / denom 
    beta = 1 - alpha

    if alpha >= -tol and beta >= -tol: # ensures alpha, beta >=0
        g_hat = alpha*p2 + beta*p1
        if np.linalg.norm(g_hat - g) <= tol:
            return np.array([beta, alpha])  # weights for [p1, p2]
    return None


# ---------------------------
#  Barycentric for 3 points
# ---------------------------

def barycentric_triangle(g, p1, p2, p3, tol=1e-12):

    """
    Function that verifies whether the point g lies 
    in the triangle made by the points p1, p2 and p3 
    and returns the coefficients u,v,w such that
    g = u*p1 + v*p2 + w*p3 if it is the case.
    """

    g = np.asarray(g)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)

    A = np.column_stack([p1 - p3, p2 - p3])
    b = g - p3

    try:
        uv = np.linalg.solve(A, b) # solves A*[u,v] = b
    except np.linalg.LinAlgError:
        return None

    u, v = uv
    w = 1 - u - v # ensures u + v+ w = 1

    if u >= -tol and v >= -tol and w >= -tol: # ensures u,v,w >= 0
        g_hat = u*p1 + v*p2 + w*p3
        if np.linalg.norm(g_hat - g) <= tol:
            return np.array([u, v, w])
    return None





def find_threshold_on_single_roc(fprs, tprs, thresholds, gamma, tol=1e-12, max_combinations=100000):
    """
    Finds thresholds corresponding to a convex combination of 1, 2, or 3 ROC points
    that exactly equals gamma (within tol), and returns the combination that
    minimizes the sum of distances between gamma and the points in the combination.

    Returns:
        selected_thresholds
        alpha (convex weights)
        gamma_hat
        subset_indices
    """

    N = len(fprs)
    points = np.column_stack((fprs, tprs))
    gamma = np.asarray(gamma)

    # compute distances
    distances = np.linalg.norm(points - gamma, axis=1)
    sorted_indices = np.argsort(distances)

    best_score = np.inf
    best_combination = None

    # 1-point combinations
    for i in sorted_indices:
        if np.linalg.norm(points[i] - gamma) <= tol:
            score = distances[i]
            if score < best_score:
                best_score = score
                best_combination = (np.array([thresholds[i]]), np.array([1.0]), gamma.copy(), (i,))

    # 2-point combinations
    count = 0
    for i, j in itertools.combinations(sorted_indices, 2):
        alpha = barycentric_segment(gamma, points[i], points[j], tol)
        if alpha is not None:
            score = distances[i] + distances[j]
            if score < best_score:
                best_score = score
                best_combination = (np.array([thresholds[i], thresholds[j]]),
                                    alpha, gamma.copy(), (i, j))
        count += 1
        if count >= max_combinations:
            break

    # 3-point combinations
    count = 0
    for i, j, k in itertools.combinations(sorted_indices, 3):
        alpha = barycentric_triangle(gamma, points[i], points[j], points[k], tol)
        if alpha is not None:
            score = distances[i] + distances[j] + distances[k]
            if score < best_score:
                best_score = score
                best_combination = (np.array([thresholds[i], thresholds[j], thresholds[k]]),
                                    alpha, gamma.copy(), (i, j, k))
        count += 1
        if count >= max_combinations:
            break

    if best_combination is not None:
        return best_combination

    raise ValueError("No exact convex combination of 1, 2, or 3 ROC points matches gamma.")
