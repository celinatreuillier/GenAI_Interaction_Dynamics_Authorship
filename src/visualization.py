import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm


def plot_barycenters_per_feature(barycenters, feature_names, n_clusters_optimal, save_fig=False):    
    palette = sns.color_palette("colorblind")
    my_palette = [palette[0], palette[1], palette[2], palette[3], palette[6], palette[9]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    labels_fig = ['# GenAI calls', '# Modified suggestions', 
                  '# Rejected suggestions', '#Accepted suggestions (without modification)']

    cluster_colors = my_palette

    for feature_idx, feature_name in enumerate(feature_names):
        ax = axes[feature_idx]
        
        for cluster_id in range(n_clusters_optimal):
            barycenter = barycenters[cluster_id]
            feature_values = barycenter[:, feature_idx]
            
            # Find valid (non-NaN) values
            valid_mask = ~np.isnan(feature_values)
            time_steps = np.arange(len(feature_values))[valid_mask]
            
            ax.plot(time_steps, feature_values[valid_mask], 
                    label=f'Cluster C{cluster_id}', 
                    linewidth=3, 
                    marker='o', 
                    markersize=6,
                    color=cluster_colors[cluster_id],
                    alpha=0.85)
        
        ax.set_xlabel('Time Window (0-31)', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title(labels_fig[feature_idx], 
                     fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(loc='best', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlim(-0.5, 31.5)

    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('../figures/barycenters_clusters.png')


def plot_calibration_scores(data, color_map, name_fig, save_fig=False):
    fig = plt.figure(figsize=(10, 8))
    calibration_values = data.iloc[:, [2]].values
    plt.scatter(
        data['written_by_human_decla'],  # X-axis: Declared Authorship
        data['written_by_human'],       # Y-axis: Real Authorship
        c=calibration_values,    # Color by calibration score
        cmap=color_map,                # Use the specified color palette
        s=70,                           # Size of the dots
        alpha=0.8,                      # Transparency for better visualization
        edgecolor='black'               # Black edges for the dots
    )

    plt.plot(
        [0, 100],  # x-coordinates
        [0, 100],  # y-coordinates
        color='gray', linestyle='--', linewidth=2, label='x = y'
    )



    norm = Normalize(vmin=calibration_values.min(), vmax=calibration_values.max())
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=color_map), ax=plt.gca())
    cbar.set_label('Authorship Calibration Score', fontsize=12)

    # Customize the plot
    # plt.title('Scatter Plot of Declared vs Real Authorship by Calibration Score', fontsize=14, fontweight='bold')
    plt.xlabel('Declared Authorship', fontsize=12)
    plt.ylabel('Real Authorship', fontsize=12)
    plt.xlim(-2, 102)
    plt.ylim(-2, 102)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Show the plot
    plt.show()

    if save_fig:
        fig.savefig('../figures/calibration_plot_{}.png'.format(name_fig))



def plot_lmm_coeffs(data, df_lmm, clusters, df_posthoc, authorship_scores, n_clusters_optimal, name_fig, save_fig=False):
    palette = sns.color_palette("colorblind")
    my_palette = [palette[0], palette[1], palette[2], palette[3], palette[6], palette[9]]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))

    sns.boxplot(
        data=data,
        x=authorship_scores,
        y="cluster",
        order=clusters,
        palette=my_palette,
        boxprops=dict(alpha=0.5),
        ax=ax,
    )

    model_means = df_lmm["mean"].tolist()
    ci_lower = df_lmm["ci_lower"].tolist()
    ci_upper = df_lmm["ci_upper"].tolist()


    yerr = [
        [m - l for m, l in zip(model_means, ci_lower)],
        [u - m for m, u in zip(model_means, ci_upper)],
    ]


    ax.errorbar(
        x=model_means,
        y=clusters,
        xerr=yerr,
        fmt="o",
        color="black",
        markersize=8,
        capsize=5,
        label="LMM Model-Estimated Mean (95% CI)",
    )

    if name_fig=='raw':
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Perfect Calibration")

    sig_pairs = df_posthoc[df_posthoc["Significant (alpha=0.05)"]].copy().reset_index(drop=True)

    if not sig_pairs.empty:
        cluster_to_y = {str(cluster): idx for idx, cluster in enumerate(clusters)}

        for idx, row in sig_pairs.iterrows():
            comparison = str(row["Comparison"]).strip()
            if " vs " not in comparison:
                continue

            left_cluster, right_cluster = [c.strip() for c in comparison.split(" vs ", 1)]
            y1 = cluster_to_y.get(str(left_cluster))
            y2 = cluster_to_y.get(str(right_cluster))

            if y1 is None or y2 is None:
                continue

            y_min = min(y1, y2) #- 0.15
            y_max = max(y1, y2) #+ 0.15
            x_pos = 0.66 + idx * 0.04
            bar_width = 0.01

            # Vertical line connecting the two clusters
            ax.plot([x_pos, x_pos], [y_min, y_max], "k-", linewidth=1.5, alpha=0.7, clip_on=False)

            # Small horizontal bars at the ends
            ax.plot([x_pos - bar_width, x_pos + bar_width], [y_min, y_min], "k-", linewidth=2, alpha=0.7, clip_on=False)
            ax.plot([x_pos - bar_width, x_pos + bar_width], [y_max, y_max], "k-", linewidth=2, alpha=0.7, clip_on=False)

            p_val = float(row["p_value_bonferroni"])
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = ""

            if stars:
                y_mid = (y_min + y_max) / 2
                ax.text(
                    x_pos + 0.02,
                    y_mid,
                    stars,
                    fontsize=10,
                    fontweight="bold",
                    va="center",
                    ha="center",
                    rotation=90,
                    clip_on=False,
                )

        ax.set_xlim(left=min(ci_lower) - 0.2, right=max(1.1, 0.9 + len(sig_pairs) * 0.06 + 0.2))


    ax.set_ylabel("Cluster", fontsize=14)
    ax.set_yticks(range(n_clusters_optimal))
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(-1, n_clusters_optimal+0.5)
    ax.set_xlabel("Authorship Calibration Score ({})".format(name_fig), fontsize=14)
    if name_fig=='raw':
        ax.set_xlim(-0.62, 0.62)
    else: 
        ax.set_xlim(-0.02, 0.62)
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    sns.despine()
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('../figures/clusters_authorship_{}.png'.format(name_fig))


def plot_time_evolution(data, model,  authorship_scores, name_fig, save_fig=False):
    fig = plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")

    sns.scatterplot(
        data=data, 
        x='session_num', 
        y=authorship_scores, 
        alpha=0.3,
        color='darkgray', 
        label='Individual sessions'
    )

    for writer, group in data.groupby('worker_id'):
        plt.plot(group['session_num'], group[authorship_scores], color='blue', alpha=0.08, linewidth=1)

    intercept = model.params['Intercept']
    slope = round(model.params['session_num'], 3)

    x_vals = np.linspace(data['session_num'].min(), data['session_num'].max(), 100)
    y_vals = intercept + slope * x_vals

    plt.plot(
        x_vals, 
        y_vals, 
        color='darkred', 
        linewidth=2.5, 
        label='Linear LMM trend'
    )


    plt.xlabel("Session Number", fontsize=11, labelpad=10)
    plt.ylabel("Calibration Score", fontsize=11, labelpad=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', loc="lower right")
    sns.despine() 

    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('../figures/evolution_authorship_{}.png'.format(name_fig))