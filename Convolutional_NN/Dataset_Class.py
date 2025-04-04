import torch
from torch.utils.data import Dataset
import os

#%%

def load_dataset(base_path='.', folder='datasets', dataset_type=None):
    """
    Carga un dataset .pth desde una carpeta, por defecto el dataset base completo.
    
    Parámetros:
    - base_path: ruta base (por defecto, carpeta actual)
    - folder: subcarpeta donde están los archivos (por defecto, 'datasets')
    - dataset_type: 'train', 'test', 'val' o None (por defecto carga el dataset base completo)
    """
    if dataset_type is None:
        filename = 'PCB_transient_dataset.pth'
    else:
        valid_types = ['train', 'test', 'val']
        if dataset_type not in valid_types:
            raise ValueError(f"Tipo de dataset inválido. Usa uno de: {valid_types} o None para el dataset base.")
        filename = f"PCB_transient_dataset_{dataset_type}.pth"

    full_path = os.path.join(base_path, folder, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {full_path}")

    print(f"✅ Cargando dataset {'base' if dataset_type is None else dataset_type} desde: {full_path}")
    return torch.load(full_path)


#%%
# Dataset class for the PCB data
# This class is used to create a dataset for the PCB data. It takes the input and output data, normalizes it, and returns it in a format that can be used by the model.
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
    
    def denormalize_input(self, x):
        """
        Desnormaliza el input completo (3 canales) usando las funciones individuales.
        Entrada: x de forma (..., 3, 13, 13)
        Salida: tensor desnormalizado con los 3 canales
        """
        device = x.device
        x_denorm = torch.empty_like(x)

        x_denorm[..., 0, :, :] = self.denormalize_T_interfaces(x[..., 0, :, :].to(device))
        x_denorm[..., 1, :, :] = self.denormalize_Q_heaters(x[..., 1, :, :].to(device))
        x_denorm[..., 2, :, :] = self.denormalize_T_env(x[..., 2, :, :].to(device))

        return x_denorm
    
    def create_input_from_values(self, Q_heaters, T_interfaces, T_env, sequence_length=1001):
        """
        Crea un input normalizado de forma (1, T, 3, 13, 13) a partir de:
        - Q_heaters: np.array de shape (4,)
        - T_interfaces: np.array de shape (4,)
        - T_env: float o escalar
        
        Devuelve: tensor (1, T, 3, 13, 13)
        """
        nodes_side = 13
    
        # Convertir a tensores
        Q_heaters = torch.tensor(Q_heaters, dtype=torch.float32)
        T_interfaces = torch.tensor(T_interfaces, dtype=torch.float32)
        T_env = torch.tensor(T_env, dtype=torch.float32)
    
        # Normalizar
        Q_norm = (Q_heaters - self.Q_heaters_mean) / self.Q_heaters_std
        T_int_norm = (T_interfaces - self.T_interfaces_mean) / self.T_interfaces_std
        T_env_norm = (T_env - self.T_env_mean) / self.T_env_std
    
        # Crear mapas (3, 13, 13)
        Q_map = torch.zeros((13, 13))
        T_map = torch.zeros((13, 13))
        T_env_map = torch.full((13, 13), T_env_norm)
    
        # Posicionar los valores
        T_map[0, 0] = T_int_norm[0]
        T_map[0, -1] = T_int_norm[1]
        T_map[-1, -1] = T_int_norm[2]
        T_map[-1, 0] = T_int_norm[3]
    
        Q_map[6, 3] = Q_norm[0]
        Q_map[3, 6] = Q_norm[1]
        Q_map[9, 3] = Q_norm[2]
        Q_map[9, 9] = Q_norm[3]
    
        # Stack y replicar en el tiempo
        input_tensor = torch.stack([T_map, Q_map, T_env_map], dim=0)  # (3, 13, 13)
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(1)         # (1, 1, 3, 13, 13)
        input_tensor = input_tensor.repeat(1, sequence_length, 1, 1, 1)  # (1, T, 3, 13, 13)
    
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
    def __init__(self, base_dataset: Dataset, max_samples: int = None, time_steps_input: int = None, time_steps_output: int = None):
        self.base_dataset = base_dataset
        self.max_samples = max_samples or len(base_dataset)
        self.time_steps_input = time_steps_input
        self.time_steps_output = time_steps_output

    def __len__(self):
        return min(self.max_samples, len(self.base_dataset))

    def __getitem__(self, idx):
        data = self.base_dataset[idx]

        if isinstance(data, tuple) and len(data) > 2:
            input_data, output_data, *bcs = data
        else:
            input_data, output_data = data
            bcs = []

        # Cortar inputs si tienen dim temporal
        if self.time_steps_input is not None and input_data.ndim >= 4:
            input_data = input_data[:self.time_steps_input]

        # Cortar outputs si tienen dim temporal
        if self.time_steps_output is not None and output_data.ndim >= 3:
            output_data = output_data[:self.time_steps_output]

        return (input_data, output_data, *bcs) if bcs else (input_data, output_data)


# -----------------------------------------------------------------------------
# load_trimmed_dataset: Carga y recorta dataset desde .pth
# -----------------------------------------------------------------------------
def load_trimmed_dataset(base_path='.', folder='datasets', dataset_type=None,
                         max_samples=None, time_steps_input=None, time_steps_output=None,
                         to_device=False):
    if dataset_type is None:
        filename = 'PCB_transient_dataset.pth'
    else:
        valid_types = ['train', 'test', 'val']
        if dataset_type not in valid_types:
            raise ValueError(f"Tipo de dataset inválido. Usa uno de: {valid_types} o None.")
        filename = f"PCB_transient_dataset_{dataset_type}.pth"

    full_path = os.path.join(base_path, folder, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {full_path}")

    print(f"✅ Cargando dataset {'base' if dataset_type is None else dataset_type} desde: {full_path}")
    base_dataset = torch.load(full_path)

    # Mover a dispositivo si se pide
    if to_device and hasattr(base_dataset, 'to_device'):
        base_dataset.to_device()
        print("📦 Dataset movido a:", "CUDA" if torch.cuda.is_available() else "CPU")

    return TrimmedDataset(base_dataset, max_samples=max_samples,
                          time_steps_input=time_steps_input,
                          time_steps_output=time_steps_output)


# -----------------------------------------------------------------------------
# prepare_data_for_convlstm: ajusta input/output para entrenamiento ConvLSTM
# -----------------------------------------------------------------------------
def prepare_data_for_convlstm(dataset, device='cuda'):
    """
    Prepara input y output para modelo ConvLSTM, asumiendo que el recorte ya se hizo en load_trimmed_dataset.
    - input: (N, C, H, W) → (N, T, C, H, W) replicando en tiempo
    - output: (N, T, H, W) → (N, T, 1, H, W)
    """
    x_list = []
    y_list = []
    for i in range(len(dataset)):
        xi, yi = dataset[i]
        x_list.append(xi)
        y_list.append(yi)
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)


    # Añadir canal a y si es necesario
    if y.ndim == 4:
        y = y.unsqueeze(2)
    elif y.ndim != 5:
        raise ValueError(f"[ERROR] Forma inesperada para y: {y.shape}")

    T = y.shape[1]  # número de pasos temporales
    x = x.unsqueeze(1).repeat(1, T, 1, 1, 1)  # (N, T, C, H, W)

    # print(f"[DEBUG] Final x: {x.shape} | y: {y.shape}")
    return x.to(device), y.to(device)
