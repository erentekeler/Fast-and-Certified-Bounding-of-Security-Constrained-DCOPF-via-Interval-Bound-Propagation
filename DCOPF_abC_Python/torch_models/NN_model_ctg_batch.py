import torch
from torch import nn
import numpy as np
import pandas as pd

from DCOPF_abC_Python.data_parser import ParseParameters


class PowerSystemsModel(nn.Module): # translation of the Gurobi model
    def __init__(self, static_params_name, ts_params_name, parameter_dtype, tensor_dtype, time_index=None):
        super().__init__()

        # Registering the tensor as a buffer so that it is moved to the same device as the model
        self.tensor_dtype = tensor_dtype

        # I am calling my dataParser here to get my model parameters
        self.parser = ParseParameters(static_params_name=static_params_name,
                                      ts_params_name=ts_params_name,
                                      dtype = parameter_dtype)

        self.time_index = time_index

        # Getting the the parameters as batches if time index is not specified
        if time_index is not None:
            duration, _, sector_costs, sector_powers = self.parser.getAllTsParamsbyTimeIndex(time_index=time_index)
            self.register_buffer('duration', torch.tensor([duration]))
        else:
            duration, _, self.sector_costs, self.sector_powers = self.parser.getTsParams()
            self.register_buffer('duration', torch.tensor(duration['duration'].values))
        
        # Getting the network parameters
        N_d2inj, Yb, Yflow, E, pf_max_base, ptdf = self.parser.getNetworkParams(from_julia=True)

        # Converting all network parameters to Torch tensors
        self.register_buffer('N_d2inj', torch.tensor(N_d2inj, dtype=self.tensor_dtype))
        self.register_buffer('Yb', torch.tensor(Yb, dtype=self.tensor_dtype))
        self.register_buffer('Yflow', torch.tensor(Yflow, dtype=self.tensor_dtype))
        self.register_buffer('E', torch.tensor(E, dtype=self.tensor_dtype))
        self.register_buffer('pf_max_base', torch.tensor(pf_max_base, dtype=self.tensor_dtype))
        self.register_buffer('ptdf', torch.tensor(ptdf, dtype=self.tensor_dtype))

        # Pushing contingency parameters to the register buffer
        M, b, u, pf_max_ctg = self.parser.getContingencyParams()
        self.register_buffer('M', torch.tensor(M, dtype=self.tensor_dtype))
        self.register_buffer('b', torch.tensor(b, dtype=self.tensor_dtype))
        self.register_buffer('u', torch.tensor(u, dtype=self.tensor_dtype))
        self.register_buffer('pf_max_ctg', torch.tensor(pf_max_ctg, dtype=self.tensor_dtype))

        # Getting the violation parameters
        self.s_flow_violation, self.p_bus_violation = self.parser.getViolationPenalties()

        device_types = self.parser.getDeviceTypes() # getting the device types for sign assignments
        self.register_buffer('sign', torch.tensor([-1 if dev_type == 'producer' else 1 for dev_type in device_types])) # sign assignment for the cost function

        # Getting the the cost parameters as batches if time index is not specified
        if time_index is not None:
            self.register_buffer('sector_cst', sector_costs)
            self.register_buffer('cum_power', torch.cumsum(sector_powers, dim=2)) # getting the x marks
            self.register_buffer('cum_cost', torch.cumsum(sector_powers * sector_costs, dim=2)) # getting the y marks
        else:
            sector_cst, sector_p = self.parser.getCostTsParamsasBatches()
            self.register_buffer('sector_cst', sector_cst)

            self.register_buffer('cum_power', torch.cumsum(sector_p, dim=2)) # getting the x marks
            self.register_buffer('cum_cost', torch.cumsum(sector_p * sector_cst, dim=2)) # getting the y marks

    

    def cst_curve(self, x): # modeling the piecewise linear cost function generically
        recursive_term = 0
        for term_idx in range(1,self.cum_cost.shape[2]):
            if term_idx==1:
                recursive_term = torch.min(self.sector_cst[:,:,term_idx] * torch.relu(x), self.cum_cost[:,:,term_idx])
            elif term_idx==self.cum_cost.shape[2]-1:
                recursive_term = recursive_term + self.sector_cst[:,:,term_idx] * torch.relu(x - self.cum_power[:,:,term_idx-1])
            else:
                recursive_term = torch.min(recursive_term + self.sector_cst[:,:,term_idx] * torch.relu(x - self.cum_power[:,:,term_idx-1]), self.cum_cost[:,:,term_idx])
        return self.sign[None, None, :] * self.duration[None, :, None] * recursive_term



    def forward(self, dev_power): # dev_power will be the bounded tensor, includes pg,and pd as a stacked vector
        pinj = torch.matmul(dev_power, self.N_d2inj) # defining the power injection at each node
        
        # Getting the device costs passing the device powers as a batch
        dev_cost = self.cst_curve(dev_power) 

        # Base case computations
        # Line flow violations are elementwise computed 
        line_flow_violation = torch.max(torch.abs(pinj @ self.ptdf) - self.pf_max_base, torch.zeros_like(self.pf_max_base))

        # computing power balance violations
        power_balance_violation = torch.abs(pinj @ (self.ptdf @ self.E.T) - pinj)

        # Base power flows
        pf_base = torch.matmul(pinj, self.ptdf)

        # Series of operations preserve the batch dimension, which is a must for bound computations
        # First contingency term
        pf_ctg_1 = pf_base[:, None, :, :] * self.M.T[None, :, None, :]

        # Second contingency term
        x = torch.matmul(pinj, self.u.T)
        pf_ctg_2 = (x[:, :, :, None] * self.b.T[None, None, :, :]).permute(0, 2, 1, 3) 

        # Computing the contingency flows
        pf_ctg = pf_ctg_1 - pf_ctg_2

        # Contingency line flow violations
        ctg_line_flow_violation = torch.relu(torch.abs(pf_ctg) - self.pf_max_ctg)

        # Objective (keeps batch first; sum over ctg and lines to stay consistent with original)
        market_surplus = (
            torch.sum(dev_cost, dim=2)
            - self.duration[None, :] * self.s_flow_violation * torch.sum(line_flow_violation, dim=2)
            - self.duration[None, :] * self.p_bus_violation * torch.sum(power_balance_violation, dim=2)
            - self.duration[None, :] * self.s_flow_violation * torch.sum(ctg_line_flow_violation, dim=(1, 3))
        )

        # output is the market surplus
        return market_surplus
    