import numpy as np
from utils import sigmoid
from logistic_regression import LogisticRegression

def train_logistic_regression(X_train, y_train):
    # Create and train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model