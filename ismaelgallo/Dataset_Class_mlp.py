import torch
from torch.utils.data import Dataset, TensorDataset
import os
import numpy as np

#%%

def load_dataset_mlp(base_path='.', folder='datasets', dataset_type=None, solver='transient'):
    """
    Carga un dataset .pth desde una carpeta, por defecto el dataset base completo.
    
    Parámetros:
    - base_path: ruta base (por defecto, carpeta actual)
    - folder: subcarpeta donde están los archivos (por defecto, 'datasets')
    - dataset_type: 'train', 'test', 'val' o None (por defecto carga el dataset base completo)
    - solver: 'transient' o 'steady' (por defecto 'transient')
    """
    if dataset_type is None:
        filename = f'PCB_mlp_{solver}_dataset.pth'
    else:
        valid_types = ['train', 'test', 'val']
        if dataset_type not in valid_types:
            raise ValueError(f"Tipo de dataset inválido. Usa uno de: {valid_types} o None para el dataset base.")
        filename = f"PCB_mlp_{solver}_dataset_{dataset_type}.pth"

    full_path = os.path.join(base_path, folder, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {full_path}")

    print(f"✅ Cargando mlp {solver} dataset {'base' if dataset_type is None else dataset_type} desde: {full_path}")
    return torch.load(full_path)


#%%
# -----------------------------------------------------------------------------
# Dataset class for the PCB data
# This class is used to create a dataset for the PCB data. It takes the input and output
# data, normalizes it, and returns it in a format that can be used by the model.
# (adapted for the mlp model)
# -----------------------------------------------------------------------------
class PCBDataset_mlp(Dataset):
    def __init__(self,T_interfaces:torch.tensor,Q_heaters:torch.tensor,T_env:torch.tensor, time:torch.tensor,T_outputs:torch.tensor,
                 T_interfaces_mean:torch.tensor,T_interfaces_std:torch.tensor,Q_heaters_mean:torch.tensor,
                 Q_heaters_std:torch.tensor,T_env_mean:torch.tensor,T_env_std:torch.tensor, time_mean:torch.tensor, 
                 time_std:torch.tensor, T_outputs_mean:torch.tensor, T_outputs_std:torch.tensor,
                 return_bc:bool = False):
        
        # print("T_interfaces shape: ", T_interfaces.shape)
        
        self.return_bc = return_bc
        
        # self.size_input = 9
        self.size_output = T_outputs.size()
        
        self.T_interfaces_mean = T_interfaces_mean
        self.T_interfaces_std = T_interfaces_std
        self.Q_heaters_mean = Q_heaters_mean
        self.Q_heaters_std = Q_heaters_std
        self.T_outputs_mean = T_outputs_mean
        self.T_outputs_std = T_outputs_std
        self.T_env_mean = T_env_mean
        self.T_env_std = T_env_std
        self.time_mean = time_mean
        self.time_std = time_std

        self.inputs = torch.empty([10]) # 4 + 4 + 1 + 1 = 10
        
        # print("T_interfaces shape: ", T_interfaces.shape)

        # print(T_interfaces_mean.shape)
        # print(T_interfaces_std.shape)
        
        self.T_interfaces = (T_interfaces-T_interfaces_mean)/T_interfaces_std
        self.Q_heaters = (Q_heaters-Q_heaters_mean)/Q_heaters_std
        self.T_env= (T_env-T_env_mean)/T_env_std
        self.time = (time-time_mean)/time_std
        self.outputs = (T_outputs-T_outputs_mean)/T_outputs_std
        
        self.inputs = torch.empty((T_interfaces.shape[0], 10), dtype=torch.float32)
        
        self.inputs[..., :4] = self.T_interfaces
        self.inputs[..., 4:8] = self.Q_heaters
        self.inputs[..., 8] = self.T_env
        self.inputs[..., 9] = self.time 
        
    def denormalize_T_interfaces(self,x):
        tensor_device = x.device
        mean = self.T_interfaces_mean.to(tensor_device)
        std = self.T_interfaces_std.to(tensor_device)
        return x*std+mean
    
    def denormalize_time(self,x):
        tensor_device = x.device
        mean = self.time_mean.to(tensor_device)
        std = self.time_std.to(tensor_device)
        return x*std+mean
    
    def denormalize_input(self, x):
        """
        Desnormaliza el input completo (9 inputs) usando las funciones individuales.
        Entrada: x de forma (..., 9)
        Salida: tensor desnormalizado con los 9 inputs
        """
        device = x.device
        x_denorm = torch.empty_like(x)

        x_denorm[:4] = self.denormalize_T_interfaces(x[:4].to(device))
        x_denorm[4:8] = self.denormalize_Q_heaters(x[4:8].to(device))
        x_denorm[8] = self.denormalize_T_env(x[8].to(device))
        x_denorm[9] = self.denormalize_time(x[9].to(device))

        return x_denorm
    
    def create_input_from_values(self, Q_heaters, T_interfaces, T_env, time, sequence_length=1001):
        """
        Crea un input normalizado de forma (9) a partir de:
        - Q_heaters: np.array de shape (4)
        - T_interfaces: np.array de shape (4)
        - T_env: float o escalar
        - time: float o escalar
        - sequence_length: longitud de la secuencia (por defecto 1001)
        
        Devuelve un tensor normalizado de forma (9)
"""
        
        # Convertir a tensores
        Q_heaters = torch.tensor(Q_heaters, dtype=torch.float32)
        T_interfaces = torch.tensor(T_interfaces, dtype=torch.float32)
        T_env = torch.tensor(T_env, dtype=torch.float32)
        time = torch.tensor(time, dtype=torch.float32)

        # Normalizar
        Q_norm = (Q_heaters - self.Q_heaters_mean) / self.Q_heaters_std
        T_int_norm = (T_interfaces - self.T_interfaces_mean) / self.T_interfaces_std
        T_env_norm = (T_env - self.T_env_mean) / self.T_env_std
        time_norm = (time - self.time_mean) / self.time_std
        
        # crear tensor de entrada (10)
        input_tensor = torch.empty((10), dtype=torch.float32)
        input_tensor[:4] = T_int_norm
        input_tensor[4:8] = Q_norm
        input_tensor[8] = T_env_norm
        input_tensor[9] = time_norm
        

        return input_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

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
            time = self.denormalize_time(self.time[idx])
            return input_data, output_data, q_heat, t_int, t_env, time

        return input_data, output_data
    
    def to_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)
        
    
#%%    

# -----------------------------------------------------------------------------
# TrimmedDataset: Wrapper para recorte eficiente de muestras y tiempo
# -----------------------------------------------------------------------------
class TrimmedDataset_mlp(Dataset):
    def __init__(self, base_dataset: Dataset, max_cases: int = None, time_steps_per_case: int = 1001):
        """
        Dataset recortado para MLP, cada muestra es un paso temporal individual.

        Args:
            base_dataset (Dataset): dataset base que retorna muestras individuales ordenadas por casos y tiempo.
            max_cases (int): número máximo de casos a devolver.
            time_steps_per_case (int): número de pasos temporales por caso.
        """
        self.base_dataset = base_dataset
        self.time_steps_per_case = time_steps_per_case
        total_cases = len(base_dataset) // time_steps_per_case
        self.max_cases = max_cases or total_cases
        self.length = self.max_cases * self.time_steps_per_case

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.base_dataset[idx]


def load_trimmed_dataset_mlp(base_path='.', folder='datasets', dataset_type=None,
                         max_cases=None, time_steps_per_case=1001,
                         to_device=False, solver='transient'):
    """
    Carga un dataset base y lo recorta en número de casos para MLP (cada muestra es un paso temporal individual).

    Args:
        base_path (str): ruta base donde está la carpeta del dataset.
        folder (str): subcarpeta donde están los archivos .pth.
        dataset_type (str): 'train', 'test', 'val' o None para archivo genérico.
        max_cases (int): número máximo de casos a devolver.
        time_steps_per_case (int): número de pasos temporales por caso en el dataset base.
        to_device (bool): si mover el dataset base a GPU si tiene método `to_device()`.
        solver (str): 'transient' o 'steady'.

    Returns:
        TrimmedDataset_mlp: dataset recortado.
    """
    # Determinar nombre de archivo
    if dataset_type is None:
        filename = f'PCB_mlp_{solver}_dataset.pth'
    else:
        valid = ['train', 'test', 'val']
        if dataset_type not in valid:
            raise ValueError(f"Tipo de dataset inválido: {dataset_type}. Usar uno de {valid}")
        filename = f'PCB_mlp_{solver}_dataset_{dataset_type}.pth'

    full_path = os.path.join(base_path, folder, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No se encontró el archivo: {full_path}")

    print(f"Cargando dataset base desde: {full_path}")
    base_dataset = torch.load(full_path)
    if isinstance(base_dataset, tuple):
        base_dataset = TensorDataset(base_dataset[0], base_dataset[1])

    if to_device and hasattr(base_dataset, 'to_device'):
        base_dataset.to_device()
        print("Dataset movido a GPU/CPU según disponibilidad")

    return TrimmedDataset_mlp(
        base_dataset=base_dataset,
        max_cases=max_cases,
        time_steps_per_case=time_steps_per_case
    )