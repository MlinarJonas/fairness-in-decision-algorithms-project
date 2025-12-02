import matplotlib.pyplot as plt

def plot_single_roc_curve(fpr, tpr, gamma=None):
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_grouped_roc_curves(fpr, tpr, labels_group):
    
    for fpr, tpr, label in zip(fpr, tpr, labels_group):
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()




