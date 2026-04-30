import pandas as pd

def perform_speed_up_analysis():
    df = pd.read_excel("all_results.xlsx", sheet_name="Benchmarked")

    sub_df = df[["n_of_buses", "Speedup"]]

    # Compute min, mean, max per number of buses
    min_df = sub_df.groupby("n_of_buses").min().reset_index().rename(columns={"Speedup":"min_speedup"})
    mean_df = sub_df.groupby("n_of_buses").mean().reset_index().rename(columns={"Speedup":"mean_speedup"})
    max_df = sub_df.groupby("n_of_buses").max().reset_index().rename(columns={"Speedup":"max_speedup"})

    # Merge all three DataFrames
    merged_df = pd.merge(min_df, mean_df, on="n_of_buses")
    merged_df = pd.merge(merged_df, max_df, on="n_of_buses")

    print(round(merged_df,2))



