import numpy as np
import matplotlib.pyplot as plt
from data_processing import get_data
from data_split import train_val_test_split
from logistic_regression import LogisticRegression


# Load and preprocess data
df = get_data()

# Split data into train, validation, and test sets
train, val, test = train_val_test_split(df)
X_train = train.drop(columns=['pass_bar']).to_numpy()
y_train = train['pass_bar'].to_numpy()

# thresholds for ROC curve
thresholds = np.linspace(0, 1, 1000)

# Train logistic regression model
model = LogisticRegression() 
model.fit(y_train, X_train)

#calculate TPR and FPR for different thresholds
tprs = []
fprs = []
for threshold in thresholds:
    y_pred = model.predict(X_train, threshold=threshold)
    tp = np.sum((y_pred == 1) & (y_train == 1))
    fp = np.sum((y_pred == 1) & (y_train == 0))
    fn = np.sum((y_pred == 0) & (y_train == 1))
    tn = np.sum((y_pred == 0) & (y_train == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    tprs.append(tpr)
    fprs.append(fpr)

# Convert to numpy arrays
tprs = np.array(tprs)
fprs = np.array(fprs)

# plot ROC curve
plt.figure()
plt.plot(fprs, tprs, label='Logistic Regression ROC')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()







