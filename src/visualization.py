import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scores_distribution(df, color, name_fig, save_fig=False):
    fig = plt.figure(figsize=(8, 3)) 

    sns.histplot(df, bins=30, color=color, kde=True)

    plt.xlabel("Authorship Calibration Score")
    plt.ylabel("# Sessions")
    plt.xlim(-0.6, 0.6)
    plt.show()

    if save_fig:
        fig.savefig('../figures/distribution_{}.png'.format(name_fig))

def plot_calibration_curves(df, color, name_fig, save_fig=False):
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(df['written_by_human_decla'], df['written_by_human'], 
            alpha=0.6, s=50, color=color)
    plt.plot([0, 100], [0, 100], '--', color='black', label='Ideal calibration', linewidth=2)
    plt.xlabel('Declared Authorship (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylabel('Actual Authorship (%)', fontsize=14)
    plt.yticks(fontsize=12)
    # plt.title(f'High GPT usage (n={len(high_data)})', fontsize=12)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('../figures/calibration_curve_{}.png'.format(name_fig))

def plot_heatmap(df, name_fig, save_fig=False):
    fig = plt.figure(figsize=(8, 6))
    h_high = plt.hist2d(df['written_by_human_decla'], df['written_by_human'],
                        bins=10, range=[[0, 100], [0, 100]], cmap='YlOrRd')
    plt.plot([0, 100], [0, 100], '--', color='black', label='Ideal calibration', linewidth=2)
    plt.xlabel('Declared Authorship (%)', fontsize=11)
    plt.ylabel('Real Authorship (%)', fontsize=11)
    # plt.title(f'High GPT usage (n={len(high_data)})', fontsize=12)
    plt.legend()
    plt.colorbar(label='Count')
    plt.tight_layout()
    plt.show()

    if save_fig:
        fig.savefig('../figures/heatmap_{}.png'.format(name_fig))