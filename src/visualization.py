import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib import cm

        
def plot_barycenters_per_feature(barycenters, feature_names, n_clusters_optimal, save_fig=False):    
    palette = sns.color_palette("colorblind")
    my_palette = [palette[0], palette[1], palette[2], palette[4]]

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


def plot_lmm_coeffs(data, df_lmm, clusters, df_posthoc, metric, n_clusters_optimal, x_axis_label, name_fig, save_fig=False):
    palette = sns.color_palette("colorblind")
    my_palette = [palette[0], palette[1], palette[2], palette[4]]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))

    sns.boxplot(
        data=data,
        x=metric,
        y="cluster",
        order=clusters,  # Reverse the order of clusters for better visualization
        palette=my_palette[::-1],
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
        y=clusters,  # Reverse the order of clusters for better visualization
        xerr=yerr,
        fmt="o",
        color="black",
        markersize=8,
        capsize=5,
        label="LMM Model-Estimated Mean (95% CI)",
    )

    if name_fig=='raw_calibration':
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

            if name_fig == 'raw_calibration' or name_fig == 'absolute_calibration':
                x_pos = 0.66 + idx * 0.04
            elif name_fig == 'authorship':
                x_pos = 105 + idx * 2
            else:  
                x_pos = 1.05 + idx * 0.04
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


    ax.set_ylabel("Cluster", fontsize=16)
    ax.set_yticks(
        ticks=range(n_clusters_optimal)[::-1],
        labels=[rf"$C_{{{i}}}$" for i in range(n_clusters_optimal)],
        fontsize=16
    )
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(-1, n_clusters_optimal+0.5)
    ax.set_xlabel(x_axis_label, fontsize=14)
    if name_fig=='raw_calibration':
        ax.set_xlim(-0.62, 0.62)
    elif name_fig=='authorship':
        ax.set_xlim(0, 100)
    elif name_fig=='absolute_calibration':
        ax.set_xlim(-0.02, 0.62)
    else: 
        ax.set_xlim(0, 1)
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    sns.despine()
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('../figures/boxplots_{}.png'.format(name_fig))


# def plot_scores_distribution(df, color, name_fig, save_fig=False):
#     fig = plt.figure(figsize=(8, 3)) 

#     sns.histplot(df, bins=30, color=color, kde=True)

#     plt.xlabel("Authorship Calibration Score")
#     plt.ylabel("# Sessions")
#     plt.xlim(-0.6, 0.6)
#     plt.show()

#     if save_fig:
#         fig.savefig('../figures/distribution_{}.png'.format(name_fig))

# def plot_calibration_curves(df, color, name_fig, save_fig=False):
#     fig = plt.figure(figsize=(8, 6))
#     plt.scatter(df['written_by_human_decla'], df['written_by_human'], 
#             alpha=0.6, s=50, color=color)
#     plt.plot([0, 100], [0, 100], '--', color='black', label='Ideal calibration', linewidth=2)
#     plt.xlabel('Declared Authorship (%)', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.ylabel('Actual Authorship (%)', fontsize=14)
#     plt.yticks(fontsize=12)
#     # plt.title(f'High GPT usage (n={len(high_data)})', fontsize=12)
#     plt.xlim(0, 100)
#     plt.ylim(0, 100)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     if save_fig:
#         fig.savefig('../figures/calibration_curve_{}.png'.format(name_fig))


# def plot_heatmap(df, name_fig, save_fig=False):
#     fig = plt.figure(figsize=(8, 6))
#     h_high = plt.hist2d(df['written_by_human_decla'], df['written_by_human'],
#                         bins=10, range=[[0, 100], [0, 100]], cmap='YlOrRd')
#     plt.plot([0, 100], [0, 100], '--', color='black', label='Ideal calibration', linewidth=2)
#     plt.xlabel('Declared Authorship (%)', fontsize=11)
#     plt.ylabel('Real Authorship (%)', fontsize=11)
#     # plt.title(f'High GPT usage (n={len(high_data)})', fontsize=12)
#     plt.legend()
#     plt.colorbar(label='Count')
#     plt.tight_layout()
#     plt.show()

#     if save_fig:
#         fig.savefig('../figures/heatmap_{}.png'.format(name_fig))