import matplotlib.pyplot as plt
from roc import get_roc_points

def plot_roc_curve(fpr_list, tpr_list, labels_group = None):
    # Plot ROC curves

    if labels_group is not None:
        for fpr, tpr, label in zip(fpr_list, tpr_list, labels_group):
            plt.plot(fpr, tpr, label=label)

    else:
        for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            plt.plot(fpr, tpr, label=f'Curve {i+1}')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

roc_points_gender = get_roc_points()
plot_roc_curve(
    fpr_list=[roc_points_gender[0][0], roc_points_gender[0][1]],
    tpr_list=[roc_points_gender[1][0], roc_points_gender[1][1]],
    labels_group= None
)


