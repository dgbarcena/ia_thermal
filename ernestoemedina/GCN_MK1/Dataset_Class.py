import torch
from torch.utils.data import Dataset

class PCBDataset(Dataset):
    def __init__(self,T_interfaces:torch.tensor,Q_heaters:torch.tensor,T_env:torch.tensor,T_outputs:torch.tensor,
                 T_interfaces_mean:torch.tensor,T_interfaces_std:torch.tensor,Q_heaters_mean:torch.tensor,
                 Q_heaters_std:torch.tensor,T_env_mean:torch.tensor,T_env_std:torch.tensor,T_outputs_mean:torch.tensor,
                 T_outputs_std:torch.tensor,
                 return_bc:bool = False):
        
        self.return_bc = return_bc
        
        self.T_interfaces_mean = T_interfaces_mean
        self.T_interfaces_std = T_interfaces_std
        self.Q_heaters_mean = Q_heaters_mean
        self.Q_heaters_std = Q_heaters_std
        self.T_outputs_mean = T_outputs_mean
        self.T_outputs_std = T_outputs_std
        self.T_env_mean = T_env_mean
        self.T_env_std = T_env_std

        self.inputs = torch.empty([T_interfaces.shape[0],3,13,13])
        self.T_interfaces = (T_interfaces-T_interfaces_mean)/T_interfaces_std
        self.Q_heaters = (Q_heaters-Q_heaters_mean)/Q_heaters_std
        self.T_env= (T_env-T_env_mean)/T_env_std
        self.outputs = (T_outputs-T_outputs_mean)/T_outputs_std
        self.inputs[:,0,:,:] = self.T_interfaces
        self.inputs[:,1,:,:] = self.Q_heaters
        self.inputs[:,2,:,:] = self.T_env

    def denormalize_T_interfaces(self,x):
        tensor_device = x.device
        mean = self.T_interfaces_mean.to(tensor_device)
        std = self.T_interfaces_std.to(tensor_device)
        return x*std+mean
    
    def denormalize_T_env(self,x):
        tensor_device = x.device
        mean = self.T_env_mean.to(tensor_device)
        std = self.T_env_std.to(tensor_device)
        return x*std+mean
    
    def denormalize_Q_heaters(self,x):
        tensor_device = x.device
        mean = self.Q_heaters_mean.to(tensor_device)
        std = self.Q_heaters_std.to(tensor_device)
        return x*std+mean

    def denormalize_output(self,x):
        tensor_device = x.device
        mean = self.T_outputs_mean.to(tensor_device)
        std = self.T_outputs_std.to(tensor_device)
        return x*std+mean

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]

        if self.return_bc:
            t_int = self.denormalize_T_interfaces(self.T_interfaces[idx])
            q_heat = self.denormalize_Q_heaters(self.Q_heaters[idx])
            t_env = self.denormalize_T_env(self.T_env[idx])
            return input_data, output_data, q_heat, t_int, t_env

        return input_data, output_data
    
    def to_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)