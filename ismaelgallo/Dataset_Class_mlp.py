import torch
from torch.utils.data import Dataset, TensorDataset
import os

#%%

def load_dataset(base_path='.', folder='datasets', dataset_type=None, solver='transient'):
    """
    Carga un dataset .pth desde una carpeta, por defecto el dataset base completo.
    
    Parámetros:
    - base_path: ruta base (por defecto, carpeta actual)
    - folder: subcarpeta donde están los archivos (por defecto, 'datasets')
    - dataset_type: 'train', 'test', 'val' o None (por defecto carga el dataset base completo)
    - solver: 'transient' o 'steady' (por defecto 'transient')
    """
    if dataset_type is None:
        filename = f'PCB_{solver}_dataset.pth'
    else:
        valid_types = ['train', 'test', 'val']
        if dataset_type not in valid_types:
            raise ValueError(f"Tipo de dataset inválido. Usa uno de: {valid_types} o None para el dataset base.")
        filename = f"PCB_{solver}_dataset_{dataset_type}.pth"

    full_path = os.path.join(base_path, folder, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {full_path}")

    print(f"✅ Cargando {solver} dataset {'base' if dataset_type is None else dataset_type} desde: {full_path}")
    return torch.load(full_path)


#%%
# -----------------------------------------------------------------------------
# Dataset class for the PCB data
# This class is used to create a dataset for the PCB data. It takes the input and output
# data, normalizes it, and returns it in a format that can be used by the model.
# (adapted for the mlp model)
# -----------------------------------------------------------------------------
class PCBDataset_mlp(Dataset):
    def __init__(self,T_interfaces:torch.tensor,Q_heaters:torch.tensor,T_env:torch.tensor,T_outputs:torch.tensor,
                 T_interfaces_mean:torch.tensor,T_interfaces_std:torch.tensor,Q_heaters_mean:torch.tensor,
                 Q_heaters_std:torch.tensor,T_env_mean:torch.tensor,T_env_std:torch.tensor,T_outputs_mean:torch.tensor,
                 T_outputs_std:torch.tensor,
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

        self.inputs = torch.empty([9])
        
        # print("T_interfaces shape: ", T_interfaces.shape)

        # print(T_interfaces_mean.shape)
        # print(T_interfaces_std.shape)
        
        self.T_interfaces = (T_interfaces-T_interfaces_mean)/T_interfaces_std
        self.Q_heaters = (Q_heaters-Q_heaters_mean)/Q_heaters_std
        self.T_env= (T_env-T_env_mean)/T_env_std
        self.outputs = (T_outputs-T_outputs_mean)/T_outputs_std
        
        self.inputs = torch.empty((T_interfaces.shape[0], 9), dtype=torch.float32)
        
        self.inputs[..., :4] = self.T_interfaces
        self.inputs[..., 4:8] = self.Q_heaters
        self.inputs[..., 8] = self.T_env
        
    def denormalize_T_interfaces(self,x):
        tensor_device = x.device
        mean = self.T_interfaces_mean.to(tensor_device)
        std = self.T_interfaces_std.to(tensor_device)
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

        return x_denorm
    
    def create_input_from_values(self, Q_heaters, T_interfaces, T_env, sequence_length=1001):
        """
        Crea un input normalizado de forma (9) a partir de:
        - Q_heaters: np.array de shape (4)
        - T_interfaces: np.array de shape (4)
        - T_env: float o escalar

        Devuelve: tensor (9)
        """
        
        # Convertir a tensores
        Q_heaters = torch.tensor(Q_heaters, dtype=torch.float32)
        T_interfaces = torch.tensor(T_interfaces, dtype=torch.float32)
        T_env = torch.tensor(T_env, dtype=torch.float32)

        # Normalizar
        Q_norm = (Q_heaters - self.Q_heaters_mean) / self.Q_heaters_std
        T_int_norm = (T_interfaces - self.T_interfaces_mean) / self.T_interfaces_std
        T_env_norm = (T_env - self.T_env_mean) / self.T_env_std
        
        # crear tensor de entrada (9)
        input_tensor = torch.empty((9), dtype=torch.float32)
        input_tensor[:4] = T_int_norm
        input_tensor[4:8] = Q_norm
        input_tensor[8] = T_env_norm
        

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
            return input_data, output_data, q_heat, t_int, t_env

        return input_data, output_data
    
    def to_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)
        
    
#%%    

# -----------------------------------------------------------------------------
# TrimmedDataset: Wrapper para recorte eficiente de muestras y tiempo
# -----------------------------------------------------------------------------
class TrimmedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, max_samples: int = None,
                 time_steps_input: int = None, time_steps_output: int = None,
                 solver: str = 'transient'):
        """
        Dataset recortado para datos estacionarios o transitorios.

        Args:
            base_dataset (Dataset): dataset base
            max_samples (int): número máximo de muestras a devolver
            time_steps_input (int): número de pasos temporales de entrada (solo si solver='transient')
            time_steps_output (int): número de pasos temporales de salida (solo si solver='transient')
            solver (str): 'steady' o 'transient'
        """
        self.base_dataset = base_dataset
        self.max_samples = max_samples or len(base_dataset)
        self.time_steps_input = time_steps_input
        self.time_steps_output = time_steps_output
        assert solver in ['steady', 'transient'], "solver must be 'steady' or 'transient'"
        self.solver = solver

    def __len__(self):
        return min(self.max_samples, len(self.base_dataset))

    def __getitem__(self, idx):
        if hasattr(self.base_dataset, 'return_bc') and self.base_dataset.return_bc:
            data = self.base_dataset[idx]
        else:
            original_return_bc = getattr(self.base_dataset, 'return_bc', False)
            self.base_dataset.return_bc = False
            data = self.base_dataset[idx]
            self.base_dataset.return_bc = original_return_bc

        if isinstance(data, tuple) and len(data) > 2:
            input_data, output_data, *bcs = data
        else:
            input_data, output_data = data
            bcs = []

        if self.solver == 'transient':
            # Solo recortar si estamos en modo transitorio
            if self.time_steps_input is not None and input_data.ndim >= 4:
                input_data = input_data[:self.time_steps_input]
            if self.time_steps_output is not None and output_data.ndim >= 3:
                output_data = output_data[:self.time_steps_output]
            # Nota: también podrías añadir chequeos de dimensión temporal más estrictos si lo deseas
            
        # Asegurar que el output tenga una dimensión de canal
        if output_data.ndim == 2:
            # caso (H, W) → (1, 1, H, W)
            output_data = output_data.unsqueeze(0).unsqueeze(1)
        elif output_data.ndim == 3:
            # caso (T, H, W) → (T, 1, H, W)
            output_data = output_data.unsqueeze(1)
        # si ya es 4D y la segunda dim es 1, mantenemos (T,1,H,W)

        return (input_data, output_data, *bcs) if bcs else (input_data, output_data)
    
# -----------------------------------------------------------------------------
# load_trimmed_dataset: Carga y recorta dataset desde .pth
# -----------------------------------------------------------------------------
def load_trimmed_dataset(base_path='.', folder='datasets', dataset_type=None,
                         max_samples=None, time_steps_input=None, time_steps_output=None,
                         to_device=False, solver='transient'):
    """
    Carga un dataset base y lo encapsula en un TrimmedDataset, compatible con casos transitorios y estacionarios.

    Args:
        base_path (str): ruta base donde está la carpeta del dataset
        folder (str): subcarpeta donde están los archivos .pth
        dataset_type (str): tipo de dataset ('train', 'test', 'val') o None
        max_samples (int): número máximo de muestras a cargar
        time_steps_input (int): número de pasos temporales de entrada (si es transitorio)
        time_steps_output (int): número de pasos temporales de salida (si es transitorio)
        to_device (bool): si se debe mover a CUDA (si está disponible)
        solver (str): 'transient' o 'steady'

    Returns:
        TrimmedDataset
    """
    if dataset_type is None:
        filename = f'PCB_{solver}_dataset.pth'
    else:
        valid_types = ['train', 'test', 'val']
        if dataset_type not in valid_types:
            raise ValueError(f"Tipo de dataset inválido. Usa uno de: {valid_types} o None.")
        filename = f"PCB_{solver}_dataset_{dataset_type}.pth"

    full_path = os.path.join(base_path, folder, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {full_path}")

    print(f"✅ Cargando dataset {'base' if dataset_type is None else dataset_type} desde: {full_path}")
    base_dataset = torch.load(full_path)

    if to_device and hasattr(base_dataset, 'to_device'):
        base_dataset.to_device()
        print("📦 Dataset movido a:", "CUDA" if torch.cuda.is_available() else "CPU")

    return TrimmedDataset(base_dataset,
                          max_samples=max_samples,
                          time_steps_input=time_steps_input,
                          time_steps_output=time_steps_output,
                          solver=solver)
