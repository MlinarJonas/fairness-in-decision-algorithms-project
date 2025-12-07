import matplotlib.pyplot as plt

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
        print(optimal_points)
        i = 0
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


import matplotlib.pyplot as plt

def plot_group_thresholds(evaluation_results, group_label_map=None, title=None):

    group_label = ["Men", "Women"]
    max_profit_res = evaluation_results["Max_Profit"]["optimal_threshold"]
    single_thr_res = evaluation_results["Overall"]["optimal_threshold"]
    equal_opp_res = evaluation_results["Equal_Opportunity"]["optimal_threshold"]
    equal_odds_res = evaluation_results["Equal_Odds"]["optimal_threshold"]
    demographic_parity_res = evaluation_results["Demographic_Parity"]["optimal_threshold"]

    # y-positions for rows (top to bottom)
    rows = {
        "Max_Profit": {"Men": 9, "Women":8},
        "Single_Threshold": {"Men": 7, "Women":6},
        "Equal_Opportunity": {"Men": 5, "Women":4},
        "Equal_Odds": {"Men": 3, "Women":2},
        "Demographic_Parity": {"Men": 1, "Women":0},
    }

    # Nice default labels / style per group (edit as you like)
    default_markers = ['D', 'o', 's', 'p', '^', 'v']   # diamond, circle, square, pentagon, ...
    default_colors  = ['b', 'g', 'r', 'c', 'm', 'y']

    fig, ax = plt.subplots(figsize=(6, 5))

    # plot the points
    for g in max_profit_res:
        if len(g) == 1:
            ax.plot(g[0], rows["Max_Profit"], marker=default_markers[0], color=default_colors[0], label = group_label[len(g)-1])
        if len(g) == 2:
            ax.plot(g[0], rows["Max_Profit"], marker=default_markers[0], color=default_colors[0], label=group_label[0])
            ax.plot(g[1], rows["Max_Profit"], marker=default_markers[1], color=default_colors[1], label= group_label[1])
        else:
            raise ValueError("More than 2 groups not supported in this plot.")
    
    for g in single_thr_res:
        ax.plot(g, rows["Single"], marker=default_markers[1], color=default_colors[1], label='Single Threshold')

    for g in equal_opp_res:
        ax.plot(g, rows["Equal_Opportunity"], marker=default_markers[2], color=default_colors[2], label='Equal Opportunity')

    for g in equal_odds_res:
        ax.plot(g, rows["Equal_Odds"], marker=default_markers[3], color=default_colors[3], label='Equal Odds')

    for g in demographic_parity_res:
        ax.plot(g, rows["Demographic_Parity"], marker=default_markers[4], color=default_colors[4], label='Demographic Parity')

    # set y-ticks and labels
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels(list(rows.keys()))
    ax.set_xlabel('Threshold Value')
    ax.set_title(title if title is not None else 'Group Thresholds by Fairness Criterion')
    ax.legend()
    plt.grid()
    plt.show()


import matplotlib.pyplot as plt

def plot_group_thresholds_test(evaluation_results, group_labels=None, title=None):
    
    # --- Extract per-criterion threshold lists (one entry per group) ---
    max_profit_res        = evaluation_results["Max_Profit"]["optimal_threshold"]
    single_thr_res        = evaluation_results["Overall"]["optimal_threshold"]
    equal_opp_res         = evaluation_results["Equal_Opportunity"]["optimal_threshold"]
    equal_odds_res        = evaluation_results["Equal_Odds"]["optimal_threshold"]
    demographic_parity_res = evaluation_results["Demographic_Parity"]["optimal_threshold"]

    # Ensure we know how many groups there are
    n_groups = 2
    # Group labels
    if group_labels is None:
        group_labels = [f"Group {i}" for i in range(n_groups)]

    # y-positions for the 5 rows (top to bottom, like the FICO plot)
    row_y = {
        "Max_Profit":        4,
        "Single":            3,
        "Equal_Opportunity": 2,
        "Equal_Odds":        1,
        "Demographic_Parity": 0,
    }

    row_order   = ["Max_Profit", "Single", "Equal_Opportunity", "Equal_Odds", "Demographic_Parity"]
    row_labels  = ["Max profit", "Single threshold", "Opportunity", "Equal odds", "Demography"]

    markers = ['D', 'o']          # enough for Men/Women
    colors  = ['b', 'g']

    fig, ax = plt.subplots(figsize=(6, 5))

    def plot_thr_list(xs, y, color, marker, label=None):
        """Plot either a single point or two points with a line in between."""
        if len(xs) == 1:
            ax.plot(xs[0], y, marker=marker, color=color,
                    linestyle='None', label=label)
        elif len(xs) == 2:
            x1, x2 = xs
            ax.plot([x1, x2], [y, y], '-', color=color)
            ax.plot([x1, x2], [y, y], marker=marker,
                    linestyle='None', color=color, label=label)
        else:
            raise ValueError(f"Expected 1 or 2 thresholds per group, got {len(xs)}.")

    # --- Plot for each group ---
    for gi in range(n_groups):
        label  = group_labels[gi]
        marker = markers[gi % len(markers)]
        color  = colors[gi % len(colors)]

        # Use label only once (on the first row) so legend has one entry per group
        plot_thr_list(max_profit_res[gi],
                      row_y["Max_Profit"],
                      color=color,
                      marker=marker,
                      label=label)

        plot_thr_list(single_thr_res[gi],
                      row_y["Single"],
                      color=color,
                      marker=marker)

        plot_thr_list(equal_opp_res[gi],
                      row_y["Equal_Opportunity"],
                      color=color,
                      marker=marker)

        plot_thr_list(equal_odds_res[gi],
                      row_y["Equal_Odds"],
                      color=color,
                      marker=marker)

        plot_thr_list(demographic_parity_res[gi],
                      row_y["Demographic_Parity"],
                      color=color,
                      marker=marker)

    # --- Styling to match the paper figure ---
    ax.set_yticks([row_y[k] for k in row_order])
    ax.set_yticklabels(row_labels)

    ax.set_xlabel("Threshold")
    ax.set_title(title if title is not None else "FICO score thresholds (raw)")

    # dotted horizontal lines between rows (optional, for nicer look)
    for y in [ (row_y["Max_Profit"]-0.5),
               (row_y["Single"]-0.5),
               (row_y["Equal_Opportunity"]-0.5),
               (row_y["Equal_Odds"]-0.5) ]:
        ax.axhline(y, linestyle=':', color='k', linewidth=0.5)

    ax.legend(title="Group", loc="upper left")
    ax.grid(axis='x', linestyle=':', linewidth=0.5)
    ax.set_ylim(-0.5, 4.5)

    fig.tight_layout()
    return fig, ax


        



    