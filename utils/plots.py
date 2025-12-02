import matplotlib.pyplot as plt

def plot_single_roc_curve(fpr, tpr, optimal_point = None):
    if optimal_point is not None:
        optimal_x, optimal_y = optimal_point
        plt.plot(optimal_x, optimal_y, 'ro', label='Optimal Point')
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_grouped_roc_curves(roc_points_by_group, labels_group, optimal_points_by_group = None):
    
    for g in roc_points_by_group:
        roc_data = roc_points_by_group[g]
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        plt.plot(fpr, tpr, label=f'Group {labels_group[g]} ROC')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()




