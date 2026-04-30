import torch
from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch_models.NN_model_ctg_batch import PowerSystemsModel
from DCOPF_abC_Python.data_parser import ParseParameters

import pandas as pd
import numpy as np
import time

def runAutoLirpa_ctg_batch(static_params_name, ts_params_name, batch_size, output_width, output_fn=None, write_xlsx=False, parameter_dtype=np.float32, tensor_dtype=torch.float32):
    # Initializing the data types and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter_dtype = parameter_dtype
    tensor_dtype = tensor_dtype

    # Setting the output width with the number of time indices 
    batch_size = batch_size
    output_width = output_width

    # Starting the timer here to capture the model building time along with IBP computations
    start_time = time.time() 
    parser = ParseParameters(static_params_name=static_params_name, ts_params_name=ts_params_name, dtype = parameter_dtype) # initializing the parser
    model = PowerSystemsModel(static_params_name=static_params_name, ts_params_name=ts_params_name, 
                              parameter_dtype=parameter_dtype, tensor_dtype=tensor_dtype)  # initializing the SC-DCOPF model
    model.to(device)

    # Getting the lower and upper bounds for the dev_power vector
    lower, upper = parser.getPlimsasBatches()
    lower = lower.to(device)
    upper = upper.to(device)

    # This is solely used for capturing the input dimension
    dev_power = lower # dummy

    # Wrap model with auto_LiRPA for bound computation.
    # The second parameter is for constructing the trace of the computational graph, and its content is not important.
    lirpa_model = BoundedModule(model, dev_power, bound_opts={"bound_every_node": True})
    pred = lirpa_model(dev_power)
    print(f'Model prediction: {pred}')
    
    # Creating the input interval and the bounded input
    ptb = PerturbationLpNorm(x_L=lower, x_U=upper)
    bounded_x = BoundedTensor(dev_power, ptb)

    # Computing output bounds using LiRPA for the given input lower and upper bounds.
    lb_ibp, ub_ibp = lirpa_model.compute_bounds(x=(bounded_x,), method='ibp')
    end_time = time.time() - start_time
    print('IBP Bounds')
    for i in range(batch_size):
        for j in range(output_width):
            print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
                j=j, i=i, l=lb_ibp[i][j].item(), u=ub_ibp[i][j].item()))
        print('---------------------------------------------------------')
    print()
    print('IBP took ', end_time, ' s')

    # Saving the bounds in an xlsx file
    if write_xlsx:
        with pd.ExcelWriter(f'DCOPF_abC_Python/output/{output_fn}', engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            bounds_df = pd.DataFrame({'Time Index':np.arange(1, output_width+1), 'Computation time': end_time, 'Lower Bound':lb_ibp[0].cpu(), 'Upper Bound':ub_ibp[0].cpu()})
            bounds_df.to_excel(writer, sheet_name='IBP', index=False)

