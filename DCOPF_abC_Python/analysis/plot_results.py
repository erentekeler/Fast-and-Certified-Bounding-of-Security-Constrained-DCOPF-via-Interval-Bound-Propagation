import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


def plot_benchmarked_gaps(save_path=None):
    df = pd.read_excel("all_results.xlsx", sheet_name="Benchmarked")
    
    # Remove infeasible cases
    sub_df = df.loc[:, ["n_of_buses", "min_gap", "mean_gap", "max_gap"]]

    grouped_df = sub_df.groupby("n_of_buses").agg({
        "min_gap": "min",
        "mean_gap": "mean",
        "max_gap": "max"
    })

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    # proper figsize for IEEE paper
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    x_pos = np.arange(len(grouped_df))

    # Shaded region
    ax.fill_between(
        x_pos,
        grouped_df["min_gap"],
        grouped_df["max_gap"],
        color='lightgray',
        alpha=0.4,
        label='Min–Max Range'
    )

    # Lines with smaller markers & widths
    ax.plot(x_pos, grouped_df["mean_gap"],
            marker='o', markersize=5, linewidth=1.5,
            color='C0', label='Mean Gap')

    ax.plot(x_pos, grouped_df["min_gap"],
            marker='s', markersize=4, linewidth=1.2,
            color='C1', label='Min Gap')

    ax.plot(x_pos, grouped_df["max_gap"],
            marker='^', markersize=4, linewidth=1.2,
            color='C2', label='Max Gap')

    # Axis labels
    ax.set_xlabel("Number of Buses", fontsize=12)
    ax.set_ylabel("IBP Gap (%)", fontsize=12)

    # Tick labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(grouped_df.index, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    # Legend
    ax.legend(fontsize=10, loc='upper left', frameon=False)

    plt.tight_layout()

    # Saving the figure
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()




def plot_ibp_violin_by_system_size_ieee(save_to=None):
    # Reading the data
    df_bench = pd.read_excel("all_results.xlsx", sheet_name="Benchmarked")
    df_nb_par = pd.read_excel("all_results.xlsx", sheet_name="nBenchmarked_parallel")
    df_nb_ser = pd.read_excel("all_results.xlsx", sheet_name="nBenchmarked_serial")

    # Extracting the columns to be plotted
    df_bench = df_bench[["n_of_buses", "IBP_mean_time"]]
    df_nb_par = df_nb_par[["n_of_buses", "IBP_mean_time"]]
    df_nb_ser = df_nb_ser[["n_of_buses", "IBP_mean_time"]]

    # combining dfs
    df = pd.concat([df_bench, df_nb_par, df_nb_ser], ignore_index=True)

    # Picking a proper figure size
    plt.figure(figsize=(7.2, 4.0))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    ax = sns.violinplot(
        data=df,
        x="n_of_buses",
        y="IBP_mean_time",
        cut=0,
        scale="width",
        inner="quartile",
        linewidth=1.2
    )

    # Mean per system size computation
    mean_df = df.groupby("n_of_buses")["IBP_mean_time"].mean().reset_index()
    bus_values = sorted(mean_df["n_of_buses"].unique())
    mean_values = mean_df.set_index("n_of_buses").loc[bus_values, "IBP_mean_time"]

    # Get x positions for each bus
    x_positions = np.arange(len(bus_values))

    # Plot vertical lines for each system size
    for x in np.arange(len(bus_values)):
        ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Plot mean as a connected line
    plt.plot(x_positions, mean_values, color='red', linewidth=1.5, marker='o', markersize=6, label='Mean runtime')

    # Y axis is a log scale
    plt.yscale("log")

    # labels and ticks 
    plt.xlabel("Number of Buses", fontsize=12)
    plt.ylabel("IBP Mean Runtime (s, log scale)", fontsize=12)
    plt.xticks(x_positions, [f"{v:,}" for v in bus_values], fontsize=10)  # Comma separator
    plt.yticks(fontsize=10)

    #  displaying the grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.grid(False)

    
    ax.legend(fontsize=10, title_fontsize=10, frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig(save_to, dpi=600, bbox_inches='tight')
    plt.show()


