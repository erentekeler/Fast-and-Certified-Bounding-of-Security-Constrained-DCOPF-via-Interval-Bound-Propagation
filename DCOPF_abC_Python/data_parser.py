import pandas as pd
import os
import h5py
import numpy as np
import torch 

"""Note that everything parsed from julia here are actually uses julia index structure, starts from 1, TODO I can maybe convert the data to Python convention."""

class ParseParameters:
    def __init__(self, static_params_name: str, ts_params_name: str, dtype = np.float32):
        # Generating the paths for the data files, the file names will be fixed here
        self.__network_params_path = os.path.join(f'DCOPF_abC_Python/parameters', static_params_name) 
        self.__ts_params_path = os.path.join(f'DCOPF_abC_Python/parameters', ts_params_name)
        self.dtype = dtype

        # Initalizing everything as class attributes with None inital values
        self.N_d2inj = self.Yb = self.Yflow = self.pf_max_base = self.ptdf = self.E = None
        self.s_flow_violation = self.p_bus_violation = None # Violation penalties are initialized
        self.device_types = None # initializing the variable keeping the device types
        self.duration_df = self.P_lims_df = self.sector_costs_df = self.sector_powers_df = None
        self.M = self.u = self.b = self.pf_max_ctg = None

        self.n_dev, self.num_ctg = None, None
   

        self._load_all() # When the object is initialized, all matrices and dfs will be assigned 


    def loadNetworkParams(self, from_julia=True):
        '''Julia stores matrices as column major order, don't need to be transposed here to get the same data.'''
        with h5py.File(self.__network_params_path, 'r') as file:
            self.N_d2inj = np.array(file["N_d2inj"], dtype=self.dtype) if from_julia else np.array(file["N_d2inj"], dtype=self.dtype).T
            self.Yb = np.array(file["Yb"], dtype=self.dtype) if from_julia else np.array(file["Yb"], dtype=self.dtype).T
            self.Yflow = np.array(file["Yflow"], dtype=self.dtype) if from_julia else np.array(file["Yflow"], dtype=self.dtype).T
            self.pf_max_base = np.array(file["pf_max_base"], dtype=self.dtype) if from_julia else np.array(file["pf_max_base"], dtype=self.dtype).T
            self.ptdf = np.array(file["ptdf"], dtype=self.dtype) if from_julia else np.array(file["ptdf"], dtype=self.dtype).T
            self.E = np.array(file["E"], dtype=self.dtype) if from_julia else np.array(file["E"], dtype=self.dtype).T

            self.M = np.array(file["M"], dtype=self.dtype) if from_julia else np.array(file["M"], dtype=self.dtype).T
            self.b = np.array(file["b"], dtype=self.dtype) if from_julia else np.array(file["b"], dtype=self.dtype).T
            self.u = np.array(file["u"], dtype=self.dtype) if from_julia else np.array(file["u"], dtype=self.dtype).T
            self.pf_max_ctg = np.array(file["pf_max_ctg"], dtype=self.dtype) if from_julia else np.array(file["pf_max_ctg"], dtype=self.dtype).T

            self.num_ctg = self.M.shape[1]
            
        

    def loadViolationPenalties(self):
        '''This function loads the violation penalty coefficients.'''
        with h5py.File(self.__network_params_path, 'r') as file:
            self.s_flow_violation = np.array(file["s_flow_violation"], dtype=self.dtype).item()
            self.p_bus_violation = np.array(file["p_bus_violation"], dtype=self.dtype).item()

            return self.s_flow_violation, self.p_bus_violation
        

    def loadDeviceTypes(self):
        '''This function loads the device types of all devices.'''
        with h5py.File(self.__network_params_path, 'r') as file:
            self.device_types = np.array([s.decode() for s in file["device_type"]])
            self.n_dev = len(self.device_types)
            return self.device_types
        

    def loadTsParams(self):
        '''This function loads the data for all time indices.'''
        with h5py.File(self.__ts_params_path, "r") as file:
            # Read duration data
            time_index = file["duration/time_index"]
            duration = file["duration/duration"]
            self.duration_df = pd.DataFrame({"time_index": time_index, "duration": duration}, dtype=self.dtype)

            # Read power limits
            P_time_index = file["P_lims/time_index"]
            P_dev = file["P_lims/dev"]
            P_lb = file["P_lims/p_lb"]
            P_ub = file["P_lims/p_ub"]
            self.P_lims_df = pd.DataFrame({"time_index": P_time_index, "dev": P_dev, "p_lb": P_lb, "p_ub": P_ub}, dtype=self.dtype)

            # Read sector costs
            sec_time_index = file["sector_costs/time_index"]
            sec_dev = file["sector_costs/dev"]
            sec_no = file["sector_costs/sector_no"]
            sec_cst = file["sector_costs/sector_cst"]
            self.sector_costs_df = pd.DataFrame({"time_index": sec_time_index, "dev": sec_dev, "sector_no": sec_no, "sector_cst": sec_cst}, dtype=self.dtype)

            # Read sector powers
            sec_p_time_index = file["sector_powers/time_index"]
            sec_p_dev = file["sector_powers/dev"]
            sec_p_no = file["sector_powers/sector_no"]
            sec_p = file["sector_powers/sector_p"]
            self.sector_powers_df = pd.DataFrame({"time_index": sec_p_time_index, "dev": sec_p_dev, "sector_no": sec_p_no, "sector_p": sec_p}, dtype=self.dtype)

            return self.duration_df, self.P_lims_df, self.sector_costs_df, self.sector_powers_df


    def _load_all(self):
        self.loadNetworkParams()
        self.loadTsParams()
        self.loadViolationPenalties()
        self.loadDeviceTypes()


    # Get methods without time index parameter
    def getNetworkParams(self, from_julia=True):
        '''Julia stores matrices as column major order, don't need to be transposed here to get the same data. \n
        N_d2inj, Yb, Yflow, E, pf_max_base, ptdf are returned, respectively'''

        return self.N_d2inj, self.Yb, self.Yflow, self.E, self.pf_max_base, self.ptdf
        

    def getViolationPenalties(self): # This function returns the violation penalties
        '''This function returns the violation penalty coefficients. \n
            s_flow_violation, p_bus_violation are returned, respectively'''
        
        return self.s_flow_violation, self.p_bus_violation
        

    def getDeviceTypes(self):
        '''This function returns the device types of all devices.'''

        return self.device_types
        

    def getTsParams(self):
        '''This function returns the data for all time indices. \n
            duration_df, P_lims_df, sector_costs_df, sector_powers_df are returned respectively'''
 
        return self.duration_df, self.P_lims_df, self.sector_costs_df, self.sector_powers_df


    # Get methods with time index parameter
    def getAllTsParamsbyTimeIndex(self, time_index): # to get all time series params for the specific time index
        '''Returns duration, P_lims, sector_costs, and sector_powers, respectively.'''
        def convertToTensor(dataframe, col_name): # This function works generically to convert the df to a tensor for cost and power
            dtype = torch.float32 if self.dtype == np.float32 else torch.float64
            df = dataframe.copy()
            # Getting the size of the tensor
            B = df["time_index"].nunique() # number of batches
            D = df["dev"].nunique() # number of devices
            S = df["sector_no"].nunique() # number of sectors 

            # This is to ensure that data is sorted properly
            df = df.sort_values(["time_index", "dev", "sector_no"])

            # Getting the sector costs for each time index and device
            grouped = df.groupby(["time_index", "dev"])[col_name].apply(lambda x: x.values)
            grouped = grouped.values.copy()

            # For some test cases, some devices have less sectors than the others
            # Those cases are padded with the last sector cst or power, basically forms a linear cost curve
            max_element = 0
            min_element = float('inf')
            for device_sectors in grouped:
                max_element = max(max_element, len(device_sectors))
                min_element = min(min_element, len(device_sectors))
            
            if max_element!=min_element: # if some devices have less sectors
                for idx, device_sectors in enumerate(grouped):
                    if len(device_sectors)!=max_element:
                        padded_sectors = np.zeros(max_element)
                        padded_sectors[:len(device_sectors)] = device_sectors
                        padded_sectors[len(device_sectors):] = device_sectors[-1]
                        grouped[idx] = padded_sectors

            # Grouped data is stacked horizontally, ie stacked as rows, then reshaped
            tensor = np.stack(grouped).reshape(B, D, S)
            tensor = torch.tensor(tensor, dtype=dtype) # casting the type to torch.tensor
            return tensor

        sector_costs_tensor = convertToTensor(self.sector_costs_df.loc[self.sector_costs_df["time_index"]==time_index], 'sector_cst')
        sector_powers_tensor = convertToTensor(self.sector_powers_df.loc[self.sector_powers_df["time_index"]==time_index], 'sector_p')
    
        return (self.duration_df.loc[self.duration_df["time_index"]==time_index, 'duration'].iloc[0], 
                self.P_lims_df.loc[self.P_lims_df["time_index"]==time_index],
                sector_costs_tensor, sector_powers_tensor)


    def getCostTsParamsbyTimeIndexandDevice(self, time_index, dev): # to get all time series cost params for the specific time index and device
        '''Returns P_lims, sector_costs and sector_powers, respectively.'''
        return (self.P_lims_df.loc[(self.P_lims_df["time_index"]==time_index) & (self.P_lims_df["dev"]==dev)],
                self.sector_costs_df.loc[(self.sector_costs_df["time_index"]==time_index) & (self.sector_costs_df["dev"]==dev)], 
                self.sector_powers_df.loc[(self.sector_powers_df["time_index"]==time_index) & (self.sector_powers_df["dev"]==dev)])
    

    def getCostTsParamsasBatches(self):
        '''Returns sector_costs and sector_powers as torch.tensors, respectively.'''
        def convertToTensor(dataframe, col_name): # This function works generically to convert the df to a tensor for cost and power
            dtype = torch.float32 if self.dtype == np.float32 else torch.float64
            df = dataframe.copy()
            # Getting the size of the tensor
            B = df["time_index"].nunique() # number of batches
            D = df["dev"].nunique() # number of devices
            S = df["sector_no"].nunique() # number of sectors 

            # This is to ensure that data is sorted properly
            df = df.sort_values(["time_index", "dev", "sector_no"])

            # Getting the sector costs for each time index and device
            grouped = df.groupby(["time_index", "dev"])[col_name].apply(lambda x: x.values)
            grouped = grouped.values.copy()

            # For some test cases, some devices have less sectors than the others
            # Those cases are padded with the last sector cst or power, basically forms a linear cost curve
            max_element = 0
            min_element = float('inf')
            for device_sectors in grouped:
                max_element = max(max_element, len(device_sectors))
                min_element = min(min_element, len(device_sectors))
            
            if max_element!=min_element: # if some devices have less sectors
                for idx, device_sectors in enumerate(grouped):
                    if len(device_sectors)!=max_element:
                        padded_sectors = np.zeros(max_element)
                        padded_sectors[:len(device_sectors)] = device_sectors
                        padded_sectors[len(device_sectors):] = device_sectors[-1]
                        grouped[idx] = padded_sectors

            # Grouped data is stacked horizontally, ie stacked as rows, then reshaped
            tensor = np.stack(grouped).reshape(B, D, S)
            tensor = torch.tensor(tensor, dtype=dtype) # casting the type to torch.tensor
            return tensor

        sector_costs_tensor = convertToTensor(self.sector_costs_df, 'sector_cst')
        sector_powers_tensor = convertToTensor(self.sector_powers_df, 'sector_p')

        return sector_costs_tensor, sector_powers_tensor
    

    def getPlimsasBatches(self):
        '''This function returns the power limits for all time indices as a batch structure \n
        returns lb and ub as torch.tensor, respectively'''
        dtype = torch.float32 if self.dtype == np.float32 else torch.float64
        df = self.P_lims_df.copy()
        # Getting the size of the tensor
        B = 1 # number of batches
        D = df["time_index"].nunique() # number of time indices
        S = df["dev"].nunique() # number of devices

        # This is to ensure that data is sorted properly
        df = df.sort_values(["time_index", "dev"])

        # Getting the ub and lb for each time index for each device
        grouped_ub = df.groupby(["time_index", "dev"])['p_ub'].apply(lambda x: x.values)
        grouped_lb = df.groupby(["time_index", "dev"])['p_lb'].apply(lambda x: x.values)
        
        # Grouped data is stacked horizontally, ie stacked as rows, then reshaped
        tensor_ub = np.stack(grouped_ub.values).reshape(B, D, S)
        tensor_ub = torch.tensor(tensor_ub, dtype=dtype) # casting the type to torch.tensor
        
        tensor_lb = np.stack(grouped_lb.values).reshape(B, D, S)
        tensor_lb = torch.tensor(tensor_lb, dtype=dtype) # casting the type to torch.tensor

        return tensor_lb, tensor_ub
    

    def getContingencyParams(self):
        '''This function returns the contingency parameters, A, b, u, and pf_max_ctg respectively.'''
        # Shapes are adjusted to be directly used in the model, to avoid modifications when the parameters are in the buffer

        self.b = self.b.T
        # self.b = np.expand_dims(self.b, axis=0)

        # self.u = np.expand_dims(self.u, axis=2) 

        return self.M, self.b, self.u, self.pf_max_ctg


