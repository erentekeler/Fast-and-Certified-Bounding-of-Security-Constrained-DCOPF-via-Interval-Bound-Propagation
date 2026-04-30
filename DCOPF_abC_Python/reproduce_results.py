import numpy as np
import torch
import os

from DCOPF_abC_Python.run_pipeline_CPU import exportNetworkFiles, runSanityCheck, exportNetworkInfo
from DCOPF_abC_Python.run_pipeline_GPU import runIBP, runIBPsi, runForwardPassTest, computeGaps
from DCOPF_abC_Python.analysis.merge_results import merge_results
from DCOPF_abC_Python.analysis.plot_results import plot_benchmarked_gaps, plot_ibp_violin_by_system_size_ieee
from DCOPF_abC_Python.analysis.speedup_analysis import perform_speed_up_analysis


'''This script is a demonstration of how we produced our results shown in the paper. 
   Results are highly hardware dependent, so, the computation times are not deterministic.
   We ran all the experiments on VACC(Vermont Advanced Computing Center) on H200 GPUs and AMD EPYC 7763 CPUs.'''

'''Creating the folders to store the parameters and output files'''
if "parameters" not in os.listdir("DCOPF_abC_Python"):
   os.mkdir("DCOPF_abC_Python/parameters")
if "output" not in os.listdir("DCOPF_abC_Python"):
   os.mkdir("DCOPF_abC_Python/output")

# These are the file names for the network files
parameter_dtype=np.float64
tensor_dtype=torch.float64

# Setting the datatype
dtype = 'float64' if (parameter_dtype==np.float64) & (tensor_dtype==torch.float64) else 'float32'
test_cases = os.listdir("DCOPF_abC_Julia/data")


'''CPU side of computations and network file exports'''
exportNetworkFiles(files_to_export=test_cases, frm="00003", upto="08316") #You can also explicitly give the file name as ["C3E1N00003D1_scenario_112.json"]
runSanityCheck(files_to_solve=test_cases, frm="00003", upto="00617", dtype=dtype) # Not possible to solve larger test cases, end up having OOM error
exportNetworkInfo(files_to_export=test_cases, frm="01576", upto="08316") # Exporting the network information of the cases that are not solved using Gurobi


'''GPU side of computations, IBP bound computation, forward pass and gap computation for the solved cases'''
# all 3- to 6717-bus cases are solved for all time indices in parallel
runIBP(files_to_solve=test_cases, frm="00003", upto="06717", parameter_dtype=parameter_dtype, tensor_dtype=tensor_dtype) 

# 8316-bus cases are solved in a serial way for all time indices, memory becomes an issue for parallel computations
runIBPsi(files_to_solve=test_cases, frm="08316", upto="08316", parameter_dtype=parameter_dtype, tensor_dtype=tensor_dtype)

# Performing sanity check by passing gurobi solutions to the computational graph, computing gaps between the gurobi solutions and IBP bounds 
test_cases_xlsx = os.listdir("DCOPF_abC_Python/output")
runForwardPassTest(files_to_solve=test_cases_xlsx, frm="00003", upto="00617", parameter_dtype=parameter_dtype, tensor_dtype=tensor_dtype)
computeGaps(files_to_solve=test_cases_xlsx, frm="00003", upto="00617")

'''This section is for the analysis and plot generations using the results'''
merge_results()
plot_benchmarked_gaps("benchmarked_gaps.pdf")
plot_ibp_violin_by_system_size_ieee("runtime_comparison.pdf")
perform_speed_up_analysis()
