import numpy as np

def generate_synthetic_student_data(
    N=3000,
    D=5,
    imbalance_ratio=0.7,  
    gender_bias=0.0,       
    random_seed=42
):
    """
    Generates synthetic data for testing logistic regression.
    Mimics "student success vs dropout" with optional gender as protected attribute.

    Args:
        N : int
            Number of samples.
        D : int
            Number of non-protected features.
        imbalance_ratio : float
            What fraction of students should have label 1 (“success”).
        gender_bias : float
            How much gender directly shifts success probability: positive → gender=1 more likely to succeed.
        random_seed : int
            For reproducibility.

    Returns:
        tx : matrix of features, shape (N, D+2) including intercept + gender feature at the end
        y  : vector indicating success or failure, shape (N,)
        true_w : weights of the non-protected features, shape (D,) 
    """
    np.random.seed(random_seed)

    # Generate a matrix of (unprotected) features ~ Normal(0,1)
    X = np.random.randn(N, D)

    # Protected attribute (binary gender)
    gender = np.random.binomial(1, 0.5, N)

    # True underlying weights for unprotected features
    true_w = np.array([1.2, -0.8, 0.5, 1.0, -1.3])[:D]

    # Gender coefficient causing bias in the outcome
    true_gender_w = gender_bias

    # Compute linear predictor z (score)
    z = X @ true_w + true_gender_w * gender

    # Adjust intercept such that class imbalance matches desired ratio
    intercept = -np.log((1 / imbalance_ratio) - 1)
    z += intercept

    # Sigmoid → probabilities (convert score to probabilities)
    probs = 1 / (1 + np.exp(-z))

    # Sample labels (indicating success or drop-out)
    y = np.random.binomial(1, probs, N)

    # Build final tx with intercept term and gender appended as last feature
    tx = np.column_stack([np.ones(N), X, gender])

    return tx, y, true_w
