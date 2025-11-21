import numpy as np
import matplotlib.pyplot as plt
from data_processing import get_data
from data_split import train_val_test_split
from logistic_regression import LogisticRegression
from utils import scores

# Load and preprocess data
df = get_data()

# Split data into train, validation, and test sets
train, val, test = train_val_test_split(df)

# Prepare training data
X_train = train.drop(columns=['pass_bar']).to_numpy()
y_train = train['pass_bar'].to_numpy()

X_val = val.drop(columns=['pass_bar']).to_numpy()
y_val = val['pass_bar'].to_numpy()

# We'll keep the gender column from val to split ROC curves
gender_val = val['gender'].to_numpy() 

# thresholds for ROC curve
thresholds = np.linspace(0, 1, 1000)

# Train logistic regression model
model = LogisticRegression() 
model.fit(X_train, y_train)

# Prepare lists for ROC for men and women
tprs_men, fprs_men = [], []
tprs_women, fprs_women = [], []


for threshold in thresholds:

    # Predict on validation set
    y_pred_val = model.predict(X_val, threshold=threshold)
    print(y_pred_val)

    # ---- MEN ----
    mask_men = (gender_val == 0)
    y_true_men = y_val[mask_men]
    y_pred_men = y_pred_val[mask_men]

    # Calculate Recall (TPR) and FPR for men
    _,_,recall_men,fpr_men,_ = scores(y_pred_men,y_true_men)
    
    tprs_men.append(recall_men)
    fprs_men.append(fpr_men)

    # ---- WOMEN ----
    mask_women = (gender_val == 1)
    y_true_women = y_val[mask_women]
    y_pred_women = y_pred_val[mask_women]

    # Calculate Recall (TPR) and FPR for women
    _,_,recall_women,fpr_women,_ = scores(y_pred_women,y_true_women)

    tprs_women.append(recall_women)
    fprs_women.append(fpr_women)


# Convert to numpy arrays
tprs_men = np.array(tprs_men)
fprs_men = np.array(fprs_men)
tprs_women = np.array(tprs_women)
fprs_women = np.array(fprs_women)

# Plot ROC curves
plt.figure()
plt.plot(fprs_men, tprs_men, label='Men ROC')
plt.plot(fprs_women, tprs_women, label='Women ROC')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve by Gender')
plt.legend()
plt.show()







