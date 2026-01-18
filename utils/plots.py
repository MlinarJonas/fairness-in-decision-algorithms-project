import matplotlib.pyplot as plt
import numpy as np

def plot_single_roc_curve(fpr, tpr, optimal_point = None, label = None):
    if optimal_point is not None:
        optimal_x, optimal_y = optimal_point
        plt.plot(optimal_x, optimal_y, 'ro', label='Optimal Point')
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label if label is not None else 'ROC Curve')
    plt.legend()
    plt.show()

def plot_grouped_roc_curves(roc_points, optimal_points, labels_group, fairness_label = None):
    if optimal_points is not None:
        i = 0
        if np.ndim(optimal_points) == 1:
            (optimal_x, optimal_y) = optimal_points
            plt.plot(optimal_x, optimal_y, 'o', label=f'Optimal Point Group {labels_group[0]}')
        else:
            for p in optimal_points:
                (optimal_x, optimal_y) = p
                plt.plot(optimal_x, optimal_y, 'o', label=f'Optimal Point Group {labels_group[i]}')
                i += 1


    for g in roc_points:
        roc_data = roc_points[g]
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        plt.plot(fpr, tpr, label=f'Group {labels_group[g]} ROC')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Grouped ROC Curves' + ' ' + fairness_label)
    plt.legend()
    plt.show()



def plot_all_optimal_points_roc(Thresholds_list, roc_points):
    
    # Different symbols and colors for optimal points
    markers = ['s', '+', '^', 'x']
    colors = ['r', 'b', 'y', 'k']

    # Labels for Roc curves
    labels_roc_curve = ["Men", "Women"]
    criteria_names = ["Max Profit", "Equal Opportunity", "Equal Odds", "Demographic Parity"]


    # Plot ROC curves
    for g in roc_points:
        roc_data = roc_points[g]
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        plt.plot(fpr, tpr, label=f'Group {labels_roc_curve[g]} ROC')

    # Plot optimal points for fairness criteria
    j = 0
    for f in Thresholds_list:
        if np.ndim(f) == 1:
            (optimal_x, optimal_y) = f
            plt.plot(optimal_x, optimal_y, marker=markers[j], color = colors[j], label=criteria_names[j], linestyle='None')
        else:
            for p in f:
                print(p)
                (optimal_x, optimal_y) = p
                plt.plot(optimal_x, optimal_y, marker=markers[j], color = colors[j], label=criteria_names[j], linestyle='None')
        j += 1

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves with optimal points for fairness criteria')
    plt.show()

def plot_group_thresholds(evaluation_results, group_label = None):

    group_label = ["Men", "Women"]
    max_profit_res = evaluation_results["Max_Profit"]["optimal_threshold"]
    single_thr_res = evaluation_results["Overall"]["optimal_threshold"]
    equal_opp_res = evaluation_results["Equal_Opportunity"]["optimal_threshold"]
    equal_odds_res = evaluation_results["Equal_Odds"]["optimal_threshold"]
    demographic_parity_res = evaluation_results["Demographic_Parity"]["optimal_threshold"]

    # y-positions for rows (top to bottom)
    rows = {
        "Max Profit": 4,
        "Single Threshold" : 3,
        "Equal Opportunity": 2,
        "Equal Odds" : 1,
        "Demographic Parity" : 0,
    }

    default_markers = ['D', 'o', 's', 'p', '^', 'v'] 
    default_colors  = ['b', 'g', 'r', 'c', 'm', 'y']

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot for max_profit
    for i, group in enumerate(max_profit_res):
        x_vals = group
        y_vals = [rows["Max Profit"]] * len(group)

        # Shift the second group a bit downward
        if i == 1:
            y_vals = [y - 0.2 for y in y_vals]   # choose any offset you like

        ax.plot(
            x_vals,
            y_vals,
            marker=default_markers[i],
            color=default_colors[i],    
            linestyle='-'
        )

    # Plot for single_threshold
    for i, group in enumerate(single_thr_res):
        x_vals = group
        y_vals = [rows["Single Threshold"]] * len(group)

        # Shift the second group a bit downward
        if i == 1:
            y_vals = [y - 0.2 for y in y_vals]   # choose any offset you like

        ax.plot(
            x_vals,
            y_vals,
            marker=default_markers[i],
            color=default_colors[i],   
            label = group_label[i], 
            linestyle='-'
        )
    

    # Plot for equal_opportunity
    for i, group in enumerate(equal_opp_res):
        x_vals = group
        y_vals = [rows["Equal Opportunity"]] * len(group)

        # Shift the second group a bit downward
        if i == 1:
            y_vals = [y - 0.2 for y in y_vals]   # choose any offset you like

        ax.plot(
            x_vals,
            y_vals,
            marker=default_markers[i],
            color=default_colors[i],    
            linestyle='-'
        )

    # Plot for equal_odds
    for i, group in enumerate(equal_odds_res):
        x_vals = group
        y_vals = [rows["Equal Odds"]] * len(group)

        # Shift the second group a bit downward
        if i == 1:
            y_vals = [y - 0.2 for y in y_vals]   # choose any offset you like

        ax.plot(
            x_vals,
            y_vals,
            marker=default_markers[i],
            color=default_colors[i],    
            linestyle='-'
        )

    # Plot for demographic_parity
    for i, group in enumerate(demographic_parity_res):
        x_vals = group
        y_vals = [rows["Demographic Parity"]] * len(group)

        # Shift the second group a bit downward
        if i == 1:
            y_vals = [y - 0.2 for y in y_vals]   # choose any offset you like

        ax.plot(
            x_vals,
            y_vals,
            marker=default_markers[i],
            color=default_colors[i],    
            linestyle='-'
        )

    ax.axhline(y=3.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=2.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=1.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)


    # set y-ticks and labels
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(list(rows.keys()))
    ax.set_xlim(0.25, 0.55)
    ax.set_xlabel('Threshold Value')
    ax.set_title('Group Thresholds by Fairness Criterion')
    ax.legend()
    plt.show()



