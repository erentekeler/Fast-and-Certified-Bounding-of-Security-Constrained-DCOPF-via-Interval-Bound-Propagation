import numpy as np  
import torch 
import os
import pandas as pd

# My scripts are imported here
from torch_models.DCOPF_model_ctg_si import PowerSystemsModel
from DCOPF_abC_Python.core.batch_bound_compute import runAutoLirpa_ctg_batch
from DCOPF_abC_Python.core.si_bound_compute import runAutoLirpa_ctg_si



def runIBP(files_to_solve='all', frm=None, upto=None, parameter_dtype=np.float32, tensor_dtype=torch.float32):
    '''This function runs IBP for the network files and saves them in the respective xlsx file. \n
        Solutions and computation times are reported. \n
        files_to_solve: Takes a json file or files as a list, if not specified all files in the DCOPF_abC_Julia/data directory are run.'''
    # This is to get the files in the data folder, case files are passed as an argument to the Julia export file
    cases = os.listdir("DCOPF_abC_Julia/data") if files_to_solve == 'all' else files_to_solve
    if frm is not None and upto is not None:
        sorted_cases = sortTestCases(test_cases=cases, frm=frm, upto=upto)
    else:
        sorted_cases = sortTestCases(test_cases=cases)

    for case in sorted_cases:
        print("\033[31mRunning IBP for \033[0m", case)
        # File names are created
        static_params_name = case.split('.json')[0] + '_network_params.h5'
        ts_params_name = case.split('.json')[0] + '_ts_params.h5'
        output_file_name = case.split('.json')[0] + '.xlsx'

        # Running the IBP function
        output_width = pd.read_excel(f'DCOPF_abC_Python/output/{output_file_name}', sheet_name='Net info').at[0, '# time indices']
        runAutoLirpa_ctg_batch(parameter_dtype=parameter_dtype, tensor_dtype=tensor_dtype, static_params_name=static_params_name, ts_params_name=ts_params_name, batch_size=1, output_width=output_width,
                               output_fn=output_file_name, write_xlsx=True)

        # Bunch of print statements to verbose
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%')
        print('Running IBP for', case)
        print('Writing on', output_file_name)
        print('IBP is successfully solved for', case)
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%', '\n')

    print("\033[31mIBP is solved for cases\033[0m", sorted_cases)
    print('\n', '\n', '\n', '\n', '\n', '\n')



def runIBPsi(files_to_solve='all', frm=None, upto=None, parameter_dtype=np.float32, tensor_dtype=torch.float32):
    '''This function runs IBP for the network files and saves them in the respective xlsx file. \n
        Solutions and computation times are reported. \n
        files_to_solve: Takes a json file or files as a list, if not specified all files in the DCOPF_abC_Julia/data directory are run.'''
    # This is to get the files in the data folder, case files are passed as an argument to the Julia export file
    cases = os.listdir("DCOPF_abC_Julia/data") if files_to_solve == 'all' else files_to_solve
    if frm is not None and upto is not None:
        sorted_cases = sortTestCases(test_cases=cases, frm=frm, upto=upto)
    else:
        sorted_cases = sortTestCases(test_cases=cases)

    for case in sorted_cases:
        print("\033[31mRunning IBP for \033[0m", case)
        # File names are created
        static_params_name = case.split('.json')[0] + '_network_params.h5'
        ts_params_name = case.split('.json')[0] + '_ts_params.h5'
        output_file_name = case.split('.json')[0] + '.xlsx'

        # Running the IBP function
        output_width = pd.read_excel(f'DCOPF_abC_Python/output/{output_file_name}', sheet_name='Net info').at[0, '# time indices']
        runAutoLirpa_ctg_si(parameter_dtype=parameter_dtype, tensor_dtype=tensor_dtype, static_params_name=static_params_name, ts_params_name=ts_params_name, batch_size=1, output_width=output_width,
                               output_fn=output_file_name, write_xlsx=True)

        # Bunch of print statements to verbose
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%')
        print('Running IBP for', case)
        print('Writing on', output_file_name)
        print('IBP is successfully solved for', case)
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%', '\n')

    print("\033[31mIBP is solved for cases\033[0m", sorted_cases)
    print('\n', '\n', '\n', '\n', '\n', '\n')



def runForwardPassTest(files_to_solve='all', frm=None, upto=None, parameter_dtype=np.float32, tensor_dtype=torch.float32):
    '''This function passes the found solutions to the model for the NN sanity check. \n
    files_to_solve: Takes an xlsx file or files as a list, if not specified all files in the DCOPF_abC_Python/output directory are run.'''
    # This is to get the files in the output folder, case files are passed as an argument to the Julia export file
    cases = os.listdir("DCOPF_abC_Python/output") if files_to_solve == 'all' else files_to_solve
    if frm is not None and upto is not None:
        sorted_cases = sortTestCases(test_cases=cases, frm=frm, upto=upto)
    else:
        sorted_cases = sortTestCases(test_cases=cases)


    for case in sorted_cases:
        print("\033[31mRunning Forward Pass Test for \033[0m", case)
        # File names are created
        static_params_name = case.split('.xlsx')[0] + '_network_params.h5'
        ts_params_name = case.split('.xlsx')[0] + '_ts_params.h5'

        # Passing the found input to the model for sanity check
        # Reading the solutions from the xlsx file
        time_indices = pd.read_excel(f'DCOPF_abC_Python/output/{case}', sheet_name='Net info').at[0, '# time indices']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        solutions = []
        for time_index in range(1, time_indices+1):
            model = PowerSystemsModel(static_params_name=static_params_name, ts_params_name=ts_params_name, time_index=time_index,
                                parameter_dtype=parameter_dtype, tensor_dtype=tensor_dtype) # specify the file path and data types
            input = torch.tensor(pd.read_excel(f'DCOPF_abC_Python/output/{case}', sheet_name='dev_powers').values, dtype=tensor_dtype)[time_index-1, :]
            input = input.view(1, 1, input.shape[0])

            model.to(device) # pass it to the gpu
            output = model(input.to(device)).item() # Those are supposed to be the solutions found in julia or ~=
            solutions.append(output) # adding the found solution to the solutions list for the time_index

        with pd.ExcelWriter(f'DCOPF_abC_Python/output/{case}', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            solutions_df = pd.DataFrame({'Time Index': np.arange(1, time_indices+1), 'Output': solutions})
            solutions_df.to_excel(writer, sheet_name='Forward Pass', index=False)

        # Bunch of print statements to verbose
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%')
        print('Running Forward Pass Test for', case)
        print('Writing on', case)
        print('Forward Pass is successfully solved for', case)
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%', '\n')

    print("\033[31mForward Pass is solved for cases\033[0m", cases)
    print('\n', '\n', '\n', '\n', '\n', '\n')



def computeGaps(files_to_solve='all', frm=None, upto=None):
    '''This function passes the found solutions to the model for the NN sanity check. \n
    files_to_solve: Takes an xlsx file or files as a list, if not specified all files in the DCOPF_abC_Python/output directory are run.'''
    # This is to get the files in the data folder, case files are passed as an argument to the Julia export file
    cases = os.listdir("DCOPF_abC_Python/output") if files_to_solve == 'all' else files_to_solve
    if frm is not None and upto is not None:
        sorted_cases = sortTestCases(test_cases=cases, frm=frm, upto=upto)
    else:
        sorted_cases = sortTestCases(test_cases=cases)

    for case in sorted_cases:
        print("\033[31mComputing the gaps for \033[0m", case)

        # Collecting all the solutions found by Gurobi, IBP and Forward Pass
        gurobi_sol = pd.read_excel(f'DCOPF_abC_Python/output/{case}', sheet_name='Gurobi')['Objective value']
        IBP_sol = pd.read_excel(f'DCOPF_abC_Python/output/{case}', sheet_name='IBP')['Upper Bound']
        FP_sol = pd.read_excel(f'DCOPF_abC_Python/output/{case}', sheet_name='Forward Pass')['Output']

        # Computing the solution gaps
        gaps_df = pd.DataFrame()
        gaps_df['IBP Gap (%)'] = 100*abs(IBP_sol - gurobi_sol)/abs(gurobi_sol)
        gaps_df['Forward Pass Gap (%)'] = 100*(FP_sol - gurobi_sol)/gurobi_sol

        with pd.ExcelWriter(f'DCOPF_abC_Python/output/{case}', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            gaps_df.to_excel(writer, sheet_name='Solution Gaps', index=False)

        # Bunch of print statements to verbose
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%')
        print('Computing gaps for', case)
        print('Writing on', case)
        print('Gaps are successfully computed for', case)
        print('%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%%%%%%', '\n')

    print("\033[31mGaps are computed for cases\033[0m", sorted_cases)
    print('\n', '\n', '\n', '\n', '\n', '\n')



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


