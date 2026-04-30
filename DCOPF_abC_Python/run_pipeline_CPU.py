import subprocess, json
import os


def exportNetworkFiles(files_to_export='all', frm=None, upto=None):
    '''This function exports the network files from Julia'''
    # This is to get the files in the data folder, case files are passed as an argument to the Julia export file
    cases = os.listdir("DCOPF_abC_Julia/data") if files_to_export == 'all' else files_to_export
    sorted_cases = sortTestCases(test_cases=cases, frm=frm, upto=upto)

    static_params_fns = []
    ts_params_fns = []
    print("\033[31mExporting Network Files\033[0m")
    for case in sorted_cases:
        # File names are created
        static_params_fn = case.split('.json')[0] + '_network_params.h5'
        ts_params_fn = case.split('.json')[0] + '_ts_params.h5'

        print("\033[31mExporting Network Files\033[0m", case)

        # file names are attached to the list for accessing them later
        static_params_fns.append(static_params_fn)
        ts_params_fns.append(ts_params_fn)

        # Executing the command
        command = ['julia', 'DCOPF_abC_Julia/src/simple_export.jl', case, static_params_fn, ts_params_fn]
        subprocess.run(command)

        # Bunch of print statements to verbose
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%')
        print('Exporting the data for the case', case)
        print('Static parameters are located in', case,'as', static_params_fn)
        print('Time series parameters are located in', case,'as', ts_params_fn)
        print('The data is successfully exported for the', case)
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%', '\n')

    print("\033[31mExporting is done\033[0m")
    print('\n', '\n', '\n', '\n', '\n', '\n')

    return static_params_fns, ts_params_fns



def runSanityCheck(files_to_solve='all', dtype=None, frm=None, upto=None):
    '''This function runs a sanity check for the network files in Julia and creates an xlsx file. \n
        Solutions and computation times are reported. \n
        files_to_solve: Takes a json file or files as a list, if not specified all files in the DCOPF_abC_Julia/data directory are run.'''
    # This is to get the files in the data folder, case files are passed as an argument to the Julia export file
    cases = os.listdir("DCOPF_abC_Julia/data") if files_to_solve == 'all' else files_to_solve
    if frm is not None and upto is not None:
        sorted_cases = sortTestCases(test_cases=cases, frm=frm, upto=upto)
    else:
        sorted_cases = sortTestCases(test_cases=cases)

    print("\033[31mRunning Sanity Check for\033[0m" , sorted_cases)

    # Executing the command
    args = json.dumps({"cases": sorted_cases, "dtype": dtype})
    command = ['julia', 'DCOPF_abC_Julia/src/sanity_check_w_ctg.jl', args]
    subprocess.run(command)


    print("\033[31mSanity check is completed for \033[0m", sorted_cases)
    print('\n', '\n', '\n', '\n', '\n', '\n')



def exportNetworkInfo(files_to_export='all', dtype=None, frm=None, upto=None):
    '''This function exports the network information for the test cases that can't be solved using Gurobi.'''
    # This is to get the files in the data folder, case files are passed as an argument to the Julia export file
    cases = os.listdir("DCOPF_abC_Julia/data") if files_to_export == 'all' else files_to_export
    if frm is not None and upto is not None:
        sorted_cases = sortTestCases(test_cases=cases, frm=frm, upto=upto)
    else:
        sorted_cases = sortTestCases(test_cases=cases)

    print("\033[31mExporting Network Info\033[0m")
    for case in sorted_cases:
        output_file_name = case.split('.json')[0] + '.xlsx'

        # Executing the command
        command = ['julia', 'DCOPF_abC_Julia/src/simple_export_net_info.jl', case, output_file_name]
        subprocess.run(command)

        # Bunch of print statements to verbose
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%')
        print('Exporting the network info for the case', case)
        print('The network info is successfully exported for the', case)
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%', '\n')

    print("\033[31mExporting is done\033[0m")
    print('\n', '\n', '\n', '\n', '\n', '\n')



def sortTestCases(test_cases, frm="00003", upto="23643"):
    # This function sorts the test cases based on their size s.t. small cases are handled first
    sorted_list = []
    donefr = False
    doneto = False
    substrings = ["00003", "00014", "00037", "00073", "00600", "00617", "01576", "02000", "04200", "04224", "06049", "06717", "08316", "23643"]
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
