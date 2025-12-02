import numpy as np
from models.logistic_regression import reg_logistic_regression_weighted
from utils.utils import sigmoid
from data.data_generation import generate_synthetic_student_data  # your existing function

# ---------------------------
# Parameters
# ---------------------------
N = 1000      # number of synthetic samples
D = 5         # number of features
lambda_ = 0.1
max_iters = 5000
gamma = 0.1
initial_w = np.zeros(D)
pos_weight_scale = 1.0

# ---------------------------
# Generate synthetic data
# ---------------------------
tx, y, true_w = generate_synthetic_student_data(N=N, D=D)

initial_w = np.zeros(tx.shape[1])

# ---------------------------
# Train logistic regression
# ---------------------------
w, loss = reg_logistic_regression_weighted(
    y, tx, lambda_, initial_w, max_iters, gamma, pos_weight_scale
)

# ---------------------------
# Evaluate model
# ---------------------------
# Compute predicted probabilities and binary predictions
pred_probs = sigmoid(tx @ w)
preds = pred_probs > 0.5

# Accuracy
accuracy = np.mean(preds == y)

# Correlation with true weights (direction wise)
corr = np.corrcoef(w[1:D+1], true_w)[0, 1]

# ---------------------------
# Print results
# ---------------------------
print("Final loss:", loss)
print("Accuracy on synthetic data:", accuracy)
print("Correlation with true weights:", corr)
print("Learned weights:", w)
print("True weights:", true_w)

