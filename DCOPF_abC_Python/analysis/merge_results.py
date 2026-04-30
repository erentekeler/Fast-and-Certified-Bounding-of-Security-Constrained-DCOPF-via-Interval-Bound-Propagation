import numpy as np  
import os
import pandas as pd


def sortTestCases(test_cases, frm="00003", upto="06717"):
    # This function sorts the test cases based on their size s.t. small cases are handled first
    sorted_list = []
    donefr = False
    doneto = False
    substrings = ["00003", "00014", "00037", "00073", "00600", "00617", "01576", "02000", "04200", "04224", "06049", "06717", "08316"]
    for substring in substrings:
        for test_case in test_cases:
            if substring == frm:
                donefr = True # outer loop break signal
            if substring == upto:
                doneto = True # outer loop break signal

            if donefr and (substring in test_case):
                sorted_list.append(test_case)

        if donefr&doneto:
            break

    return sorted_list



def merge_results():
    cases = os.listdir("DCOPF_abC_Python/output")
    
    # Instances from 3 bus to 617 buses are solved by Gurobi, populating results accordingly
    # Creating the dataframe 
    benchmarked_results_df = pd.DataFrame(columns=["case_name", "n_of_buses", "n_time_indices", "min_gap", "mean_gap", "max_gap", 
                                                   "IBP_total_time", "IBP_mean_time", "Gurobi_total_time", "Speedup", 
                                                   "is_all_infeasible", "is_any_infeasible"])
    benchmarked_cases = sortTestCases(test_cases=cases, frm="00003", upto="00617")

    for case in benchmarked_cases:
        print(case)
        case_name = case.split(".")[0]
        file_path = "DCOPF_abC_Python/output/" + case
        n_of_buses = pd.read_excel(file_path, sheet_name="Net info").loc[0, "# buses"]
        n_time_indices = pd.read_excel(file_path, sheet_name="Net info").loc[0, "# time indices"]

        min_gap = pd.read_excel(file_path, sheet_name="Solution Gaps")["IBP Gap (%)"].min()
        mean_gap = pd.read_excel(file_path, sheet_name="Solution Gaps")["IBP Gap (%)"].mean()
        max_gap = pd.read_excel(file_path, sheet_name="Solution Gaps")["IBP Gap (%)"].max()
        IBP_total_time = pd.read_excel(file_path, sheet_name="IBP")["Computation time"].mean()
        IBP_mean_time = IBP_total_time/n_time_indices
        Gurobi_total_time = pd.read_excel(file_path, sheet_name="Gurobi")["Computation time"].sum()
        speedup = Gurobi_total_time/IBP_total_time
        is_all_infeasible = (pd.read_excel(file_path, sheet_name="IBP")["Upper Bound"] < 0).all()
        is_any_infeasible = (pd.read_excel(file_path, sheet_name="IBP")["Upper Bound"] < 0).any()

        benchmarked_results_df.loc[len(benchmarked_results_df), :] = {"case_name":case_name, "n_of_buses":n_of_buses, "n_time_indices":n_time_indices,
                                                                      "min_gap":min_gap, "mean_gap":mean_gap, "max_gap":max_gap, 
                                                                      "IBP_total_time":IBP_total_time, "IBP_mean_time": IBP_mean_time,
                                                                      "Gurobi_total_time":Gurobi_total_time, "Speedup": speedup,
                                                                      "is_all_infeasible": is_all_infeasible, "is_any_infeasible": is_any_infeasible}

        benchmarked_results_df.to_excel("all_results.xlsx", sheet_name="Benchmarked", index=True)



    



    # Instances from 1576 bus to 6717 buses are solved only by IBP, in parallel
    # Creating the dataframe 
    non_benchmarked_parallel_results_df = pd.DataFrame(columns=["case_name", "n_of_buses", "n_time_indices", "IBP_total_time", "IBP_mean_time",
                                                                "is_all_infeasible", "is_any_infeasible"])
    non_benchmarked_parallel_cases = sortTestCases(test_cases=cases, frm="01576", upto="06717")

    for case in non_benchmarked_parallel_cases:
        print(case)
        case_name = case.split(".")[0]
        file_path = "DCOPF_abC_Python/output/" + case
        n_of_buses = pd.read_excel(file_path, sheet_name="Net info").loc[0, "# buses"]
        n_time_indices = pd.read_excel(file_path, sheet_name="Net info").loc[0, "# time indices"]

        IBP_total_time = pd.read_excel(file_path, sheet_name="IBP")["Computation time"].mean()
        IBP_mean_time = IBP_total_time/n_time_indices
        is_all_infeasible = (pd.read_excel(file_path, sheet_name="IBP")["Upper Bound"] < 0).all()
        is_any_infeasible = (pd.read_excel(file_path, sheet_name="IBP")["Upper Bound"] < 0).any()

        non_benchmarked_parallel_results_df.loc[len(non_benchmarked_parallel_results_df), :] = {"case_name":case_name, "n_of_buses":n_of_buses,
                                                                                                "n_time_indices": n_time_indices,
                                                                                                "IBP_total_time":IBP_total_time,
                                                                                                "IBP_mean_time": IBP_mean_time,
                                                                                                "is_all_infeasible": is_all_infeasible,
                                                                                                "is_any_infeasible": is_any_infeasible}


    with pd.ExcelWriter("all_results.xlsx", mode="a", engine="openpyxl",if_sheet_exists="replace") as writer:
        non_benchmarked_parallel_results_df.to_excel(writer, sheet_name="nBenchmarked_parallel", index=True)



    

    # Instances with 08316 buses are solved only by IBP, serial
    # Creating the dataframe 
    non_benchmarked_serial_results_df = pd.DataFrame(columns=["case_name", "n_of_buses", "n_time_indices", "IBP_total_time", "IBP_mean_time",
                                                              "is_all_infeasible", "is_any_infeasible"])
    non_benchmarked_serial_cases = sortTestCases(test_cases=cases, frm="08316", upto="08316")

    for case in non_benchmarked_serial_cases:
        print(case)
        case_name = case.split(".")[0]
        file_path = "DCOPF_abC_Python/output/" + case
        n_of_buses = pd.read_excel(file_path, sheet_name="Net info").loc[0, "# buses"]
        n_time_indices = pd.read_excel(file_path, sheet_name="Net info").loc[0, "# time indices"]

        IBP_total_time = pd.read_excel(file_path, sheet_name="IBP")["Computation time"].sum()
        IBP_mean_time = pd.read_excel(file_path, sheet_name="IBP")["Computation time"].mean()

        is_all_infeasible = (pd.read_excel(file_path, sheet_name="IBP")["Upper Bound"] < 0).all()
        is_any_infeasible = (pd.read_excel(file_path, sheet_name="IBP")["Upper Bound"] < 0).any()

        non_benchmarked_serial_results_df.loc[len(non_benchmarked_serial_results_df), :] = {"case_name":case_name, "n_of_buses":n_of_buses,
                                                                                            "n_time_indices": n_time_indices,
                                                                                            "IBP_total_time":IBP_total_time, 
                                                                                            "IBP_mean_time": IBP_mean_time,                                                                                     
                                                                                            "is_all_infeasible": is_all_infeasible,
                                                                                            "is_any_infeasible": is_any_infeasible}


    with pd.ExcelWriter("all_results.xlsx", mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        non_benchmarked_serial_results_df.to_excel(writer, sheet_name="nBenchmarked_serial", index=True)
